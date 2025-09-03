import json
import os
import queue
import re
import subprocess
import time
import uuid

import matplotlib.pyplot as plt
import numpy as np
from PyQt5.QtCore import Qt, QThread, QTimer
from PyQt5.QtWidgets import (
    QComboBox,
    QDialog,
    QDockWidget,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from hardware.Ulster.gui.main_window_ext.zone_measurements import ZoneMeasurementsMixin
from hardware.Ulster.gui.technical.capture import (
    CaptureWorker,
    show_measurement_window,
    validate_folder,
)
from hardware.Ulster.gui.technical.measurement_worker import MeasurementWorker


class PoniFileSelectionDialog(QDialog):
    """Dialog for selecting PONI files for each detector alias."""

    def __init__(self, aliases, current_poni_files=None, parent=None):
        super().__init__(parent)
        self.aliases = aliases
        self.poni_files = {}
        self.line_edits = {}

        # Pre-populate with current PONI files if available
        if current_poni_files:
            for alias in aliases:
                if alias in current_poni_files:
                    poni_info = current_poni_files[alias]
                    if isinstance(poni_info, dict) and "path" in poni_info:
                        self.poni_files[alias] = poni_info["path"]
                    elif (
                        hasattr(self.parent(), "poni_files")
                        and alias in self.parent().poni_files
                    ):
                        # Fallback to parent's poni_files if available
                        parent_poni = self.parent().poni_files[alias]
                        if isinstance(parent_poni, dict) and "path" in parent_poni:
                            self.poni_files[alias] = parent_poni["path"]

        self.setup_ui()

    def setup_ui(self):
        self.setWindowTitle("Select PONI Files for Technical Meta")
        self.setModal(True)
        self.resize(600, 400)

        layout = QVBoxLayout(self)

        # Header
        header = QLabel("Select PONI calibration files for each detector alias:")
        header.setStyleSheet("font-weight: bold; margin-bottom: 10px;")
        layout.addWidget(header)

        # Form layout for PONI file selection
        form_layout = QFormLayout()

        for alias in self.aliases:
            # Create horizontal layout for each alias
            h_layout = QHBoxLayout()

            # Line edit for file path
            line_edit = QLineEdit()
            line_edit.setPlaceholderText(f"Select PONI file for {alias}")
            if alias in self.poni_files:
                line_edit.setText(self.poni_files[alias])
            self.line_edits[alias] = line_edit
            h_layout.addWidget(line_edit)

            # Browse button
            browse_btn = QPushButton("Browse...")
            browse_btn.clicked.connect(
                lambda checked, a=alias: self.browse_poni_file(a)
            )
            h_layout.addWidget(browse_btn)

            # Clear button
            clear_btn = QPushButton("Clear")
            clear_btn.clicked.connect(lambda checked, a=alias: self.clear_poni_file(a))
            h_layout.addWidget(clear_btn)

            form_layout.addRow(f"{alias}:", h_layout)

        layout.addLayout(form_layout)

        # Buttons
        button_layout = QHBoxLayout()

        ok_btn = QPushButton("OK")
        ok_btn.clicked.connect(self.accept)
        button_layout.addWidget(ok_btn)

        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        button_layout.addWidget(cancel_btn)

        layout.addStretch()
        layout.addLayout(button_layout)

    def browse_poni_file(self, alias):
        """Open file dialog to select PONI file for the given alias."""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            f"Select PONI File for {alias}",
            "",
            "PONI Files (*.poni);;All Files (*)",
        )
        if file_path:
            self.line_edits[alias].setText(file_path)
            self.poni_files[alias] = file_path

    def clear_poni_file(self, alias):
        """Clear the PONI file selection for the given alias."""
        self.line_edits[alias].setText("")
        if alias in self.poni_files:
            del self.poni_files[alias]

    def get_poni_files(self):
        """Return dictionary of alias -> poni_file_path."""
        result = {}
        for alias, line_edit in self.line_edits.items():
            path = line_edit.text().strip()
            if path:
                result[alias] = path
        return result


class TechnicalMeasurementsMixin(ZoneMeasurementsMixin):

    NO_SELECTION_LABEL = "— Select —"
    TYPE_OPTIONS = ["AGBH", "DARK", "EMPTY", "BACKGROUND"]

    def create_technical_panel(self):
        self.aux_counter = 0
        super().create_zone_measurements()

        self.measDock = QDockWidget("Technical Measurements", self)
        self.measDock.setAllowedAreas(Qt.LeftDockWidgetArea | Qt.RightDockWidgetArea)

        container = QWidget()
        outer = QVBoxLayout(container)
        outer.setContentsMargins(8, 8, 8, 8)
        outer.setSpacing(12)

        # Integration time control
        it_layout = QHBoxLayout()
        it_layout.addWidget(QLabel("Integration Time (s):"))
        self.integrationTimeSpin = QDoubleSpinBox()
        self.integrationTimeSpin.setRange(0.1, 1e4)
        self.integrationTimeSpin.setSingleStep(0.1)
        self.integrationTimeSpin.setValue(1.0)
        it_layout.addWidget(self.integrationTimeSpin)
        outer.addLayout(it_layout)

        # Save folder selector
        fld = QHBoxLayout()
        fld.addWidget(QLabel("Save Folder:"))
        self.folderLE = QLineEdit()
        default_folder = (
            self.config.get("default_folder", "") if hasattr(self, "config") else ""
        )
        self.folderLE.setText(default_folder)

        fld.addWidget(self.folderLE, 1)
        b = QPushButton("Browse…")
        b.clicked.connect(self._browse_folder)
        fld.addWidget(b)
        outer.addLayout(fld)

        # Auxiliary Measurement controls
        outer.addWidget(QLabel("Aux Measurement:"))
        row = QHBoxLayout()
        self.auxBtn = QPushButton("Measure Aux")
        self.auxBtn.clicked.connect(self.measure_aux)
        row.addWidget(self.auxBtn)

        self._aux_status = QLabel("")
        row.addWidget(self._aux_status)
        self._aux_timer = QTimer(self)
        self._aux_timer.setInterval(200)
        self._aux_timer.timeout.connect(self._update_aux_status)

        self.auxNameLE = QLineEdit()
        self.auxNameLE.setPlaceholderText("Name for Aux Measurement")
        row.addWidget(self.auxNameLE, 1)
        outer.addLayout(row)

        # Aux measurements table
        self.auxTable = QTableWidget()
        self.auxTable.setColumnCount(3)
        self.auxTable.setHorizontalHeaderLabels(
            [
                "File",
                "Type",
                "Alias",
            ]
        )
        # Stretch columns
        try:
            from PyQt5.QtWidgets import QHeaderView

            self.auxTable.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        except Exception:
            pass
        self.auxTable.setSelectionBehavior(self.auxTable.SelectRows)
        self.auxTable.setSelectionMode(self.auxTable.ExtendedSelection)
        self.auxTable.cellDoubleClicked.connect(self._open_measurement_from_table)
        outer.addWidget(self.auxTable)

        # Aux table actions
        aux_actions = QHBoxLayout()
        load_btn = QPushButton("Load Files…")
        load_btn.setToolTip("Load existing technical measurement files into the table")
        load_btn.clicked.connect(self.load_technical_files)
        aux_actions.addWidget(load_btn)
        outer.addLayout(aux_actions)

        # PyFai button
        pyfai_btn = QPushButton("PyFai")
        pyfai_btn.setToolTip("Run pyfai-calib2 in this folder")
        pyfai_btn.clicked.connect(self.run_pyfai)
        outer.addWidget(pyfai_btn)

        # Generate Meta button
        gen_btn = QPushButton("Generate Meta")
        gen_btn.setToolTip("Generate technical_meta_*.json from selected rows")
        gen_btn.clicked.connect(self.generate_technical_meta)
        outer.addWidget(gen_btn)

        # Real-time controls
        rt_layout = QHBoxLayout()
        rt_layout.addWidget(QLabel("Frames/⟳:"))
        self.framesSpin = QSpinBox()
        self.framesSpin.setRange(1, 1_000_000)
        self.framesSpin.setValue(1)
        rt_layout.addWidget(self.framesSpin)

        self.rtBtn = QPushButton("Real-time")
        self.rtBtn.setCheckable(True)
        self.rtBtn.clicked.connect(self._toggle_realtime)
        rt_layout.addWidget(self.rtBtn)

        outer.addLayout(rt_layout)

        # Wrap in a scroll area and add to dock
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(container)
        self.measDock.setWidget(scroll)
        self.addDockWidget(Qt.LeftDockWidgetArea, self.measDock)

        self.enable_measurement_controls(False)
        self.hardware_state_changed.connect(self.enable_measurement_controls)
        # Refresh alias models when hardware state changes
        self.hardware_state_changed.connect(
            lambda _: self.refresh_aux_table_alias_models()
        )

    def enable_measurement_controls(self, enable: bool):
        widgets = [
            self.integrationTimeSpin,
            self.folderLE,
            self.auxBtn,
            self.auxNameLE,
            self.auxTable,
            self.framesSpin,
            self.rtBtn,
        ]
        for w in widgets:
            w.setEnabled(enable)

    def _browse_folder(self):
        f = QFileDialog.getExistingDirectory(self, "Select Folder")
        if f:
            self.folderLE.setText(f)

    def _start_capture(self, typ: str):
        counter_attr = f"{typ.lower()}_counter"
        count = getattr(self, counter_attr, 0) + 1
        setattr(self, counter_attr, count)

        folder = validate_folder(self.folderLE.text())
        base = self._file_base(typ)
        base_with_count = f"{base}_{count:03d}"
        ts = time.strftime("%Y%m%d_%H%M%S")
        txt_filename_base = os.path.join(
            folder,
            f"{base_with_count}_{ts}_{int(self.integrationTimeSpin.value())}s",
        )

        worker = CaptureWorker(
            detector_controller=self.detector_controller,
            integration_time=self.integrationTimeSpin.value(),
            txt_filename_base=txt_filename_base,
        )
        thread = QThread()
        worker.moveToThread(thread)
        thread.started.connect(worker.run)  # .run() method needed in worker

        def _cleanup(success, result_files, t=typ):
            try:
                self._on_capture_done(success, result_files, t)
            except Exception as e:
                print(f"Error in _on_capture_done for {t}: {e}")
            finally:
                worker.deleteLater()
                thread.quit()
                thread.deleteLater()
                self._capture_workers.remove(worker)

        worker.finished.connect(_cleanup)
        thread.start()

        if not hasattr(self, "_capture_workers"):
            self._capture_workers = []
        self._capture_workers.append(worker)

    def _on_capture_done(self, success: bool, result_files: dict, typ: str):
        if not success:
            print(f"[{typ}] capture failed.")
            self._aux_timer.stop()
            self._aux_status.setText("")
            return

        print(f"[{typ}] capture successful: {result_files}")
        self._aux_timer.stop()
        self._aux_status.setText("Processing...")

        # --- Set up worker
        worker = MeasurementWorker(
            filenames=result_files,
        )
        worker.add_aux_item.connect(self._add_aux_item_to_list)
        worker.run()

    def _add_aux_item_to_list(self, alias, npy_path):
        """Add a new row to the Aux table with file, type and alias selectors.
        Also validates filename format: name_timestamp_..._ALIAS.ext (timestamp before alias).
        """
        from pathlib import Path

        from PyQt5.QtCore import Qt

        # Validate naming: ensure timestamp before alias
        try:
            if not self._validate_timestamp_before_alias(npy_path):
                from PyQt5.QtWidgets import QMessageBox

                QMessageBox.warning(
                    self,
                    "Filename format",
                    "File name should include timestamp before detector alias\n"
                    "Expected pattern like: name_YYYYMMDD_HHMMSS_..._ALIAS.ext",
                )
        except Exception:
            pass

        row = self.auxTable.rowCount()
        self.auxTable.insertRow(row)

        # File column (read-only, store full path in UserRole)
        display = f"{alias}: {Path(npy_path).name}"
        file_item = QTableWidgetItem(display)
        file_item.setFlags(Qt.ItemIsSelectable | Qt.ItemIsEnabled)
        file_item.setData(Qt.UserRole, str(npy_path))
        self.auxTable.setItem(row, 0, file_item)

        # Type combobox (blank by default)
        type_cb = self._make_type_combobox()
        self.auxTable.setCellWidget(row, 1, type_cb)

        # Alias combobox (preselect the source alias)
        alias_cb = self._make_alias_combobox(preselect=alias)
        self.auxTable.setCellWidget(row, 2, alias_cb)

    def _file_base(self, typ: str) -> str:
        le: QLineEdit = getattr(self, f"{typ.lower()}NameLE")
        txt = le.text().strip().replace(" ", "_")
        return txt or typ.lower()

    # ---- Upload and validation helpers ----
    def _validate_timestamp_before_alias(self, file_path: str) -> bool:
        """Return True if the file name has a timestamp (YYYYMMDD_HHMMSS) before alias.
        Accepts names like: name_YYYYMMDD_HHMMSS_..._ALIAS.ext"""
        base = os.path.basename(file_path)
        # Remove extension
        name, _ext = os.path.splitext(base)
        # Expect at least 3 tokens separated by underscores
        toks = name.split("_")
        if len(toks) < 3:
            return False
        # Look for timestamp token 'YYYYMMDD_HHMMSS' across two tokens or combined with underscore
        # Our generator uses a single token with embedded '_': YYYYMMDD_HHMMSS
        stamp_match = re.search(r"\d{8}_\d{6}", name)
        if not stamp_match:
            return False
        # Ensure alias is the last token
        alias = toks[-1]
        # Minimal alias check: alphanumeric
        if not re.fullmatch(r"[A-Za-z0-9]+", alias):
            return False
        # Ensure the timestamp appears before the alias
        return stamp_match.start() < (len(name) - len(alias))

    def _infer_alias_from_filename(self, file_path: str) -> str:
        base = os.path.basename(file_path)
        alias = os.path.splitext(base)[0].split("_")[-1]
        # Validate against active aliases if available
        try:
            active_aliases = self._get_active_detector_aliases()
            if alias in active_aliases:
                return alias
        except Exception:
            pass
        return alias  # fallback

    def load_technical_files(self):
        """Load existing technical measurement files into the aux table.
        Validates file naming and tries to infer alias from the filename."""
        from PyQt5.QtWidgets import QFileDialog, QMessageBox

        files, _ = QFileDialog.getOpenFileNames(
            self,
            "Load Technical Measurement Files",
            str(self.folderLE.text() or ""),
            "NumPy Arrays (*.npy);;Text Files (*.txt);;All Files (*)",
        )
        if not files:
            return

        for fpath in files:
            # Convert .txt to .npy next to it (non-destructive)
            path_to_use = fpath
            try:
                if fpath.lower().endswith(".txt"):
                    data = np.loadtxt(fpath)
                    npy_path = os.path.splitext(fpath)[0] + ".npy"
                    np.save(npy_path, data)
                    path_to_use = npy_path
            except Exception as e:
                QMessageBox.warning(
                    self,
                    "Conversion failed",
                    f"Failed to convert TXT to NPY for:\n{fpath}\nError: {e}",
                )
                continue

            # Validate filename format
            if not self._validate_timestamp_before_alias(path_to_use):
                QMessageBox.warning(
                    self,
                    "Filename format",
                    "File name should include timestamp before detector alias\n"
                    "Expected pattern like: name_YYYYMMDD_HHMMSS_..._ALIAS.ext",
                )
                # Continue adding anyway, but user is informed

            alias = self._infer_alias_from_filename(path_to_use)
            self._add_aux_item_to_list(alias, path_to_use)

    # ---- Persist/restore aux table in global state ----
    def build_aux_state(self):
        """Serialize current auxTable rows to a list for state saving."""
        rows = []
        try:
            if not hasattr(self, "auxTable") or self.auxTable is None:
                return rows
            for r in range(self.auxTable.rowCount()):
                file_item = self.auxTable.item(r, 0)
                file_path = (
                    file_item.data(Qt.UserRole) if file_item is not None else None
                )
                # Type
                type_cb = self.auxTable.cellWidget(r, 1)
                type_text = None
                try:
                    if type_cb is not None:
                        t = type_cb.currentText()
                        if t and t != self.NO_SELECTION_LABEL:
                            type_text = t
                except Exception:
                    pass
                # Alias
                alias_cb = self.auxTable.cellWidget(r, 2)
                alias_text = None
                try:
                    if alias_cb is not None:
                        a = alias_cb.currentText()
                        if a and a != self.NO_SELECTION_LABEL:
                            alias_text = a
                except Exception:
                    pass
                rows.append(
                    {"file_path": file_path, "type": type_text, "alias": alias_text}
                )
        except Exception as e:
            print(f"Error building aux state: {e}")
        return rows

    def restore_technical_aux_rows(self, rows):
        """Restore auxTable rows from previously saved state."""
        try:
            if not hasattr(self, "auxTable") or self.auxTable is None:
                return
            # Clear existing rows
            self.auxTable.setRowCount(0)
            for row in rows or []:
                fpath = row.get("file_path")
                alias = row.get("alias") or self._infer_alias_from_filename(fpath or "")
                self._add_aux_item_to_list(alias or "", fpath or "")
                # Set type if provided
                try:
                    rix = self.auxTable.rowCount() - 1
                    type_cb = self.auxTable.cellWidget(rix, 1)
                    if type_cb is not None and row.get("type"):
                        idx = (
                            type_cb.findText(row["type"])
                            if hasattr(type_cb, "findText")
                            else -1
                        )
                        if idx >= 0:
                            type_cb.setCurrentIndex(idx)
                except Exception:
                    pass
        except Exception as e:
            print(f"Error restoring aux rows: {e}")

    def measure_aux(self):
        self._aux_start = time.time()
        self._aux_spinner_state = 0
        self._aux_status.setText("0 s ⠋")
        self._aux_timer.start()
        self._start_capture("Aux")

    def _open_measurement_from_table(self, row: int, _col: int):
        """Open measurement window for the selected row."""
        from PyQt5.QtCore import Qt

        file_item = self.auxTable.item(row, 0)
        if not file_item:
            return
        file_path = file_item.data(Qt.UserRole)

        # Prefer alias from Alias #1 if selected, else try to infer from display text
        alias_cb = self.auxTable.cellWidget(row, 2)
        alias = None
        if isinstance(alias_cb, QComboBox):
            a = alias_cb.currentText().strip()
            if a and a != self.NO_SELECTION_LABEL:
                alias = a

        if not alias:
            disp = file_item.text()
            if ":" in disp:
                alias = disp.split(":", 1)[0].strip()

        if not alias:
            # Fallback to first controller alias
            try:
                alias = next(iter(self.detector_controller))
            except Exception:
                alias = None

        show_measurement_window(
            file_path, self.masks.get(alias), self.ponis.get(alias), self
        )

    def run_pyfai(self):
        env = self.config.get("conda")
        if not env:
            print("❌ No conda env set in self.config['conda']")
            return

        folder = validate_folder(self.folderLE.text())

        if os.name == "nt":
            cmd = (
                f"CALL conda activate {env} " f'&& cd /d "{folder}" ' f"&& pyfai-calib2"
            )
            start_cmd = f'start cmd /K "{cmd}"'
            try:
                subprocess.Popen(start_cmd, shell=True)
                print("▶️ Launched PyFai in new cmd window.")
            except Exception as e:
                print("❌ Failed to launch PyFai on Windows:", e)
        else:
            bash_cmd = (
                f'cd "{folder}" && '
                f"conda activate {env} && "
                "pyfai-calib2; exec bash"
            )
            try:
                subprocess.Popen(["bash", "-lc", bash_cmd])
                print("▶️ Launched PyFai in new bash window.")
            except Exception as e:
                print("❌ Failed to launch PyFai on Unix:", e)

    def initialize_hardware(self):
        pass

    def _update_aux_status(self):
        elapsed = int(time.time() - self._aux_start)
        spinner = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
        ch = spinner[self._aux_spinner_state % len(spinner)]
        self._aux_spinner_state += 1
        self._aux_status.setText(f"{elapsed} s {ch}")

    def _toggle_realtime(self, checked: bool):
        if checked:
            self._start_realtime()
            self.rtBtn.setText("Stop RT")
        else:
            self._stop_realtime()
            self.rtBtn.setText("Real-time")

    def _start_realtime(self):
        exposure = float(self.integrationTimeSpin.value())
        self._rt_queue = queue.Queue()

        plt.ion()
        detector_aliases = list(self.detector_controller.keys())
        n_det = len(detector_aliases)
        self._rt_img = {}
        self._rt_last_frame = {}  # <--- Cache for latest frame per alias

        # One subplot per detector alias
        fig, axes = plt.subplots(1, n_det, figsize=(5 * n_det, 5))
        if n_det == 1:
            axes = [axes]

        for ax, alias in zip(axes, detector_aliases):
            size = getattr(self.detector_controller[alias], "size", (256, 256))
            self._rt_img[alias] = ax.imshow(
                np.zeros(size), origin="lower", interpolation="none"
            )
            ax.set_title(alias)
        self._rt_fig = fig
        plt.show()

        self._plot_timer = QTimer(self)
        self._plot_timer.setInterval(50)
        self._plot_timer.timeout.connect(self._rt_plot_tick)
        self._plot_timer.start()

        def callback(frames_dict):
            # Cache most recent frame per alias
            for alias, frame in frames_dict.items():
                self._rt_last_frame[alias] = frame
            self._rt_queue.put(True)  # Just a signal to the timer

        # Start stream on all detectors
        for controller in self.detector_controller.values():
            controller.start_stream(
                callback=callback, exposure=exposure, interval=0.0, frames=1
            )

    def _rt_plot_tick(self):
        # Drain the queue (we only need to plot once per timer tick)
        while True:
            try:
                _ = self._rt_queue.get_nowait()
            except queue.Empty:
                break
        # Update all subplots with their latest frame
        for alias in self._rt_img:
            frame = self._rt_last_frame.get(alias)
            if frame is not None:
                self._rt_img[alias].set_data(frame)
                self._rt_img[alias].set_clim(frame.min(), frame.max())
        self._rt_fig.canvas.draw_idle()

    def _stop_realtime(self):
        for controller in self.detector_controller.values():
            controller.stop_stream()
        if hasattr(self, "_plot_timer"):
            self._plot_timer.stop()
            del self._plot_timer
        import matplotlib.pyplot as plt

        plt.close(self._rt_fig)
        del self._rt_queue
        del self._rt_last_frame

    # -------------------- Helpers for Aux Table --------------------
    def _get_active_detector_aliases(self):
        """Return aliases from settings (main.json), honoring DEV/dev_active_detectors.
        This intentionally reads from config instead of live hardware."""
        dev_mode = self.config.get("DEV", False)
        ids = (
            self.config.get("dev_active_detectors", [])
            if dev_mode
            else self.config.get("active_detectors", [])
        )
        return [
            d.get("alias")
            for d in self.config.get("detectors", [])
            if d.get("id") in ids
        ]

    def _make_type_combobox(self):
        cb = QComboBox()
        cb.addItem(self.NO_SELECTION_LABEL, None)
        for t in self.TYPE_OPTIONS:
            cb.addItem(t, t)
        return cb

    def _make_alias_combobox(self, preselect=None):
        cb = QComboBox()
        cb.addItem(self.NO_SELECTION_LABEL, None)
        for alias in self._get_active_detector_aliases():
            cb.addItem(alias, alias)
        if preselect:
            idx = cb.findText(preselect)
            if idx >= 0:
                cb.setCurrentIndex(idx)
        return cb

    def refresh_aux_table_alias_models(self):
        aliases = self._get_active_detector_aliases()
        for row in range(self.auxTable.rowCount()):
            cb = self.auxTable.cellWidget(row, 2)
            if not isinstance(cb, QComboBox):
                continue
            current = cb.currentText()
            cb.blockSignals(True)
            cb.clear()
            cb.addItem(self.NO_SELECTION_LABEL, None)
            for a in aliases:
                cb.addItem(a, a)
            # restore selection if still present
            if current and current in aliases:
                cb.setCurrentText(current)
            cb.blockSignals(False)

    # -------------------- Generate Technical Meta --------------------
    def generate_technical_meta(self):
        from pathlib import Path

        from PyQt5.QtWidgets import QMessageBox

        # Validate selection
        sel = (
            self.auxTable.selectionModel().selectedRows()
            if self.auxTable.selectionModel()
            else []
        )
        rows = [idx.row() for idx in sel]
        if not rows:
            QMessageBox.warning(
                self, "No Selection", "Select one or more rows in the Aux table."
            )
            return

        # Validate name and folder
        name = (self.auxNameLE.text() or "").strip()
        if not name:
            QMessageBox.warning(
                self, "Missing Name", "Enter a name in the Aux Measurement field."
            )
            return
        safe_name = name.replace(" ", "_")
        folder = validate_folder(self.folderLE.text())
        if not os.path.isdir(folder):
            QMessageBox.warning(self, "Invalid Folder", "Select a valid save folder.")
            return

        out_path = os.path.join(folder, f"technical_meta_{safe_name}.json")
        if os.path.exists(out_path):
            res = QMessageBox.question(
                self,
                "Overwrite?",
                f"File exists:\n{out_path}\n\nDo you want to overwrite it?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            )
            if res != QMessageBox.Yes:
                return

        meta = {}
        seen_pairs = set()  # (type, alias)

        for row in rows:
            file_item = self.auxTable.item(row, 0)
            if not file_item:
                continue
            file_path = file_item.data(Qt.UserRole)
            if not file_path or not os.path.exists(file_path):
                QMessageBox.warning(
                    self, "Missing File", f"Row {row+1}: file path does not exist."
                )
                return

            # Type
            type_cb = self.auxTable.cellWidget(row, 1)
            if (
                not isinstance(type_cb, QComboBox)
                or type_cb.currentText() == self.NO_SELECTION_LABEL
            ):
                QMessageBox.warning(
                    self, "Missing Type", f"Row {row+1}: select measurement type."
                )
                return
            typ = type_cb.currentText()

            # Alias (must be selected)
            cb = self.auxTable.cellWidget(row, 2)
            if (
                not isinstance(cb, QComboBox)
                or cb.currentText() == self.NO_SELECTION_LABEL
            ):
                QMessageBox.warning(
                    self, "Missing Alias", f"Row {row+1}: select an alias."
                )
                return
            al = cb.currentText()

            base = os.path.basename(file_path)
            dst = meta.setdefault(typ, {})
            pair = (typ, al)
            if pair in seen_pairs or al in dst:
                QMessageBox.warning(
                    self,
                    "Duplicate Assignment",
                    f"Measurement for type '{typ}' and alias '{al}' is already assigned.",
                )
                return
            dst[al] = base
            seen_pairs.add(pair)

        # Get unique aliases from selected measurements for PONI file selection
        unique_aliases = set()
        for row in rows:
            cb = self.auxTable.cellWidget(row, 2)
            if (
                isinstance(cb, QComboBox)
                and cb.currentText() != self.NO_SELECTION_LABEL
            ):
                unique_aliases.add(cb.currentText())

        # Show PONI file selection dialog if we have aliases
        poni_lab = {}
        if unique_aliases:
            # Get current PONI files if available
            current_poni_files = getattr(self, "poni_files", {})

            poni_dialog = PoniFileSelectionDialog(
                aliases=sorted(unique_aliases),
                current_poni_files=current_poni_files,
                parent=self,
            )

            if poni_dialog.exec_() == QDialog.Accepted:
                selected_poni_files = poni_dialog.get_poni_files()
                poni_lab_path = {}
                poni_lab_values = {}

                # Process each selected PONI file
                for alias, file_path in selected_poni_files.items():
                    # Store filename for PONI_LAB
                    poni_lab[alias] = os.path.basename(file_path)

                    # Store full path for PONI_LAB_PATH
                    poni_lab_path[alias] = file_path

                    # Read and store PONI file content for PONI_LAB_VALUES
                    try:
                        with open(file_path, "r", encoding="utf-8") as f:
                            poni_content = f.read()
                            poni_lab_values[alias] = poni_content
                    except Exception as e:
                        QMessageBox.warning(
                            self,
                            "PONI File Read Error",
                            f"Failed to read PONI file for {alias}:\n{file_path}\n\nError: {e}\n\nContinuing without this PONI file content.",
                        )
                        # Still include the filename and path, but mark content as unavailable
                        poni_lab_values[alias] = (
                            f"# ERROR: Could not read PONI file content\n# File: {file_path}\n# Error: {str(e)}"
                        )

                # Store additional PONI data for later use
                self._temp_poni_lab_path = poni_lab_path
                self._temp_poni_lab_values = poni_lab_values
            else:
                # User cancelled PONI selection, ask if they want to continue without PONI files
                res = QMessageBox.question(
                    self,
                    "No PONI Files Selected",
                    "Do you want to generate the technical meta file without PONI calibration files?",
                    QMessageBox.Yes | QMessageBox.No,
                    QMessageBox.No,
                )
                if res != QMessageBox.Yes:
                    return

        # Add PONI sections to meta if any PONI files were selected
        if poni_lab:
            meta["PONI_LAB"] = poni_lab

        # Add PONI_LAB_PATH section if available
        if hasattr(self, "_temp_poni_lab_path") and self._temp_poni_lab_path:
            meta["PONI_LAB_PATH"] = self._temp_poni_lab_path

        # Add PONI_LAB_VALUES section if available
        if hasattr(self, "_temp_poni_lab_values") and self._temp_poni_lab_values:
            meta["PONI_LAB_VALUES"] = self._temp_poni_lab_values

        # Add or reuse a calibration group hash so multiple files can be grouped together
        try:
            group_hash = getattr(self, "calibration_group_hash", None)
            if not group_hash:
                group_hash = uuid.uuid4().hex[:16]
                setattr(self, "calibration_group_hash", group_hash)
            meta["CALIBRATION_GROUP_HASH"] = group_hash
        except Exception:
            # Non-fatal; proceed without the group hash if something unexpected happens
            pass

        # Write JSON
        try:
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(meta, f, indent=2)
        except Exception as e:
            QMessageBox.critical(
                self, "Write Error", f"Failed to write meta file:\n{e}"
            )
            return
        finally:
            # Clean up temporary PONI data variables
            if hasattr(self, "_temp_poni_lab_path"):
                delattr(self, "_temp_poni_lab_path")
            if hasattr(self, "_temp_poni_lab_values"):
                delattr(self, "_temp_poni_lab_values")

        # Summary
        try:
            summary_lines = []
            for k, v in meta.items():
                if isinstance(v, dict):
                    summary_lines.append(f"{k}: {len(v)} file(s)")
                else:
                    summary_lines.append(f"{k}: {v}")
            summary = "\n".join(summary_lines) or "(empty)"
        except Exception:
            summary = "(summary unavailable)"
        QMessageBox.information(
            self, "Meta Generated", f"Saved to:\n{out_path}\n\nSummary:\n{summary}"
        )
