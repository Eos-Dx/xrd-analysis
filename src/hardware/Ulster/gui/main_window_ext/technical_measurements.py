import os
import time
import numpy as np
import subprocess

from PyQt5.QtWidgets import (
    QDockWidget, QWidget, QHBoxLayout, QVBoxLayout,
    QLabel, QLineEdit, QPushButton, QFileDialog,
    QListWidget, QListWidgetItem, QDoubleSpinBox, QScrollArea
)
from PyQt5.QtCore import Qt

from hardware.Ulster.gui.technical.capture import CaptureWorker, validate_folder
from hardware.Ulster.gui.technical.capture import show_measurement_window
from hardware.Ulster.gui.main_window_ext.zone_measurements_extension import ZoneMeasurementsMixin

class TechnicalMeasurementsMixin(ZoneMeasurementsMixin):

    def create_measurements_panel(self):
        # Initialize counter for auxiliary measurements
        self.aux_counter = 0

        # Call parent widget creation if needed
        super().create_zone_measurements_widget()

        # Create technical measurements dock
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
        fld.addWidget(self.folderLE, 1)
        b = QPushButton("Browse…")
        b.clicked.connect(self._browse_folder)
        fld.addWidget(b)
        outer.addLayout(fld)

        # Auxiliary Measurement
        outer.addWidget(QLabel("Aux Measurement:"))
        row = QHBoxLayout()
        btn = QPushButton("Measure Aux")
        btn.clicked.connect(self.measure_aux)
        self.auxBtn = btn
        row.addWidget(btn)

        # --- status label for timer + spinner ---
        from PyQt5.QtCore import QTimer
        self._aux_status = QLabel("")
        row.addWidget(self._aux_status)
        # timer to tick every 200ms and update label
        self._aux_timer = QTimer(self)
        self._aux_timer.setInterval(200)
        self._aux_timer.timeout.connect(self._update_aux_status)

        le = QLineEdit()
        le.setPlaceholderText("Name for Aux Measurement")
        self.auxNameLE = le
        row.addWidget(le, 1)
        outer.addLayout(row)

        # Measurement list
        self.auxList = QListWidget()
        self.auxList.itemActivated.connect(self.open_measurement)
        outer.addWidget(self.auxList)

        # PyFai button
        pyfai_btn = QPushButton("PyFai")
        pyfai_btn.setToolTip("Run pyfai-calib2 in this folder")
        pyfai_btn.clicked.connect(self.run_pyfai)
        outer.addWidget(pyfai_btn)

        # Wrap in a scroll area so contents can scroll if needed
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(container)
        self.measDock.setWidget(scroll)
        self.addDockWidget(Qt.LeftDockWidgetArea, self.measDock)

        # Initially disable controls until hardware is initialized
        self.enable_measurement_controls(False)
        self.hardware_state_changed.connect(self.enable_measurement_controls)

    def enable_measurement_controls(self, enable: bool):
        widgets = [
            self.integrationTimeSpin, self.folderLE,
            self.auxBtn, self.auxNameLE, self.auxList
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
        txt = os.path.join(folder, f"{base_with_count}_{ts}_{int(self.integrationTimeSpin.value())}s.txt")

        worker = CaptureWorker(
            detector_controller=self.detector_controller,
            integration_time=self.integrationTimeSpin.value(),
            txt_filename=txt
        )

        if not hasattr(self, "_capture_workers"):
            self._capture_workers = []
        self._capture_workers.append(worker)

        def _cleanup(ok, fn, t=typ):
            try:
                self._on_capture_done(ok, fn, t)
            finally:
                worker.deleteLater()
                self._capture_workers.remove(worker)

        worker.finished.connect(_cleanup)
        worker.start()

    def _on_capture_done(self, success: bool, txt_file: str, typ: str):
        if not success:
            print(f"[{typ}] capture failed.")
            self._aux_timer.stop()
            self._aux_status.setText("")
            return
        else:
            print(f'[{typ}] capture successful: {txt_file}')
            self._aux_timer.stop()
            self._aux_status.setText("Done")

        try:
            data = np.loadtxt(txt_file)
            npy = txt_file.replace(".txt", ".npy")
            np.save(npy, data)
        except Exception as e:
            print("Conversion error:", e)
            npy = txt_file


        lst: QListWidget = getattr(self, f"{typ.lower()}List")
        item = QListWidgetItem(os.path.basename(npy))
        item.setData(Qt.UserRole, npy)
        lst.addItem(item)

    def _file_base(self, typ: str) -> str:
        le: QLineEdit = getattr(self, f"{typ.lower()}NameLE")
        txt = le.text().strip().replace(" ", "_")
        return txt or typ.lower()

    def measure_aux(self):
        # record start time and start status timer
        self._aux_start = time.time()
        self._aux_spinner_state = 0
        self._aux_status.setText("0 s ⠋")  # initial text
        self._aux_timer.start()
        # Start an auxiliary measurement capture
        self._start_capture("Aux")

    def open_measurement(self, item: QListWidgetItem):
        show_measurement_window(
            item.data(Qt.UserRole),
            self.mask,
            getattr(self, "poni", None),
            self
        )

    def run_pyfai(self):
        env = self.config.get("conda")
        if not env:
            print("❌ No conda env set in self.config['conda']")
            return

        folder = validate_folder(self.folderLE.text())

        if os.name == "nt":
            cmd = (
                f'CALL conda activate {env} '
                f'&& cd /d "{folder}" '
                f'&& pyfai-calib2'
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
                f'conda activate {env} && '
                'pyfai-calib2; exec bash'
            )
            try:
                subprocess.Popen(["bash", "-lc", bash_cmd])
                print("▶️ Launched PyFai in new bash window.")
            except Exception as e:
                print("❌ Failed to launch PyFai on Unix:", e)

    def initialize_hardware(self):
        # Override to subscribe automatically
        pass  # subscription to hardware_state_changed handles control toggling

    def _update_aux_status(self):
        """
        Called every 200ms to update the elapsed seconds
        and spinner character.
        """
        elapsed = int(time.time() - self._aux_start)
        # cycle through a simple 4‐step spinner
        spinner = ["⠋","⠙","⠹","⠸","⠼","⠴","⠦","⠧","⠇","⠏"]
        ch = spinner[self._aux_spinner_state % len(spinner)]
        self._aux_spinner_state += 1
        # update label: "12 s ⠼"
        self._aux_status.setText(f"{elapsed} s {ch}")