import os
import time
import numpy as np
import subprocess
import matplotlib.pyplot as plt

from PyQt5.QtWidgets import (
    QDockWidget, QWidget, QHBoxLayout, QVBoxLayout,
    QLabel, QLineEdit, QPushButton, QFileDialog,
    QListWidget, QListWidgetItem, QDoubleSpinBox,
    QSpinBox, QScrollArea
)
from PyQt5.QtCore import Qt, QTimer
import queue

from hardware.Ulster.gui.technical.capture import (
    CaptureWorker, validate_folder, show_measurement_window
)
from hardware.Ulster.gui.main_window_ext.zone_measurements_extension import ZoneMeasurementsMixin


class TechnicalMeasurementsMixin(ZoneMeasurementsMixin):

    def create_measurements_panel(self):
        self.aux_counter = 0
        super().create_zone_measurements_widget()

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

        self.auxList = QListWidget()
        self.auxList.itemActivated.connect(self.open_measurement)
        outer.addWidget(self.auxList)

        # PyFai button
        pyfai_btn = QPushButton("PyFai")
        pyfai_btn.setToolTip("Run pyfai-calib2 in this folder")
        pyfai_btn.clicked.connect(self.run_pyfai)
        outer.addWidget(pyfai_btn)

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

    def enable_measurement_controls(self, enable: bool):
        widgets = [
            self.integrationTimeSpin, self.folderLE,
            self.auxBtn, self.auxNameLE, self.auxList,
            self.framesSpin, self.rtBtn
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
        txt_filename_base = os.path.join(folder, f"{base_with_count}_{ts}_{int(self.integrationTimeSpin.value())}s")

        worker = CaptureWorker(
            detector_controller=self.detector_controller,
            integration_time=self.integrationTimeSpin.value(),
            txt_filename_base=txt_filename_base
        )

        if not hasattr(self, "_capture_workers"):
            self._capture_workers = []
        self._capture_workers.append(worker)

        def _cleanup(success, result_files, t=typ):
            try:
                self._on_capture_done(success, result_files, t)
            finally:
                worker.deleteLater()
                self._capture_workers.remove(worker)

        worker.finished.connect(_cleanup)
        worker.start()

    def _on_capture_done(self, success: bool, result_files: dict, typ: str):
        if not success:
            print(f"[{typ}] capture failed.")
            self._aux_timer.stop()
            self._aux_status.setText("")
            return
        else:
            print(f'[{typ}] capture successful: {result_files}')
            self._aux_timer.stop()
            self._aux_status.setText("Done")

        for det in ['WAXS', 'SAXS']:
            txt_file = result_files[det]

            # --- Save in subfolder ---
            det_folder = os.path.join(os.path.dirname(txt_file), det)
            os.makedirs(det_folder, exist_ok=True)
            new_txt_file = os.path.join(det_folder, os.path.basename(txt_file))
            os.replace(txt_file, new_txt_file)  # Move file to subfolder

            # Convert to .npy in same subfolder
            try:
                data = np.loadtxt(new_txt_file)
                npy = new_txt_file.replace(".txt", ".npy")
                np.save(npy, data)
            except Exception as e:
                print(f"Conversion error for {det}:", e)
                npy = new_txt_file

            # Add to auxList: show detector label
            item = QListWidgetItem(f"{det}: {os.path.basename(npy)}")
            item.setData(Qt.UserRole, npy)
            self.auxList.addItem(item)

    def _file_base(self, typ: str) -> str:
        le: QLineEdit = getattr(self, f"{typ.lower()}NameLE")
        txt = le.text().strip().replace(" ", "_")
        return txt or typ.lower()

    def measure_aux(self):
        self._aux_start = time.time()
        self._aux_spinner_state = 0
        self._aux_status.setText("0 s ⠋")
        self._aux_timer.start()
        self._start_capture("Aux")

    def open_measurement(self, item: QListWidgetItem):
        file_path = item.data(Qt.UserRole)
        # Detect which detector (assumes filename contains "WAXS" or "SAXS")
        if "WAXS" in file_path:
            detector = "WAXS"
        elif "SAXS" in file_path:
            detector = "SAXS"
        else:
            detector = "WAXS"  # fallback
        show_measurement_window(
            file_path,
            self.masks.get(detector),
            self.ponis.get(detector),
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
        pass

    def _update_aux_status(self):
        elapsed = int(time.time() - self._aux_start)
        spinner = ["⠋","⠙","⠹","⠸","⠼","⠴","⠦","⠧","⠇","⠏"]
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

        import matplotlib.pyplot as plt
        plt.ion()
        self._rt_fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
        self._rt_img = {
            "WAXS": ax1.imshow(np.zeros((256, 256)), origin='lower', interpolation='none'),
            "SAXS": ax2.imshow(np.zeros((256, 256)), origin='lower', interpolation='none')
        }
        ax1.set_title("WAXS")
        ax2.set_title("SAXS")
        plt.show()

        self._plot_timer = QTimer(self)
        self._plot_timer.setInterval(50)
        self._plot_timer.timeout.connect(self._rt_plot_tick)
        self._plot_timer.start()

        def callback(frames_dict):
            self._rt_queue.put(frames_dict)

        self.detector_controller.start_stream(
            callback=callback,
            exposure=exposure,
            interval=0.0,
            frames=1
        )

    def _rt_plot_tick(self):
        frames_dict = None
        while True:
            try:
                frames_dict = self._rt_queue.get_nowait()
            except queue.Empty:
                break
        if frames_dict is not None:
            for name in ("WAXS", "SAXS"):
                frame = frames_dict.get(name)
                if frame is not None and name in self._rt_img:
                    self._rt_img[name].set_data(frame)
                    self._rt_img[name].set_clim(frame.min(), frame.max())
            self._rt_fig.canvas.draw_idle()

    def _stop_realtime(self):
        self.detector_controller.stop_stream()
        if hasattr(self, '_plot_timer'):
            self._plot_timer.stop()
            del self._plot_timer
        import matplotlib.pyplot as plt
        plt.close(self._rt_fig)
        del self._rt_queue
