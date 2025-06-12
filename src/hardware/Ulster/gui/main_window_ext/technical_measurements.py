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
from PyQt5.QtCore import Qt
import queue
from PyQt5.QtCore import QTimer
from hardware.Ulster.gui.technical.capture import (
    CaptureWorker, validate_folder, show_measurement_window
)
from hardware.Ulster.gui.main_window_ext.zone_measurements_extension import ZoneMeasurementsMixin


class TechnicalMeasurementsMixin(ZoneMeasurementsMixin):

    def create_measurements_panel(self):
        # Initialize counter for auxiliary measurements
        self.aux_counter = 0
        super().create_zone_measurements_widget()

        # Create technical measurements dock
        self.measDock = QDockWidget("Technical Measurements", self)
        self.measDock.setAllowedAreas(Qt.LeftDockWidgetArea | Qt.RightDockWidgetArea)

        container = QWidget()
        outer = QVBoxLayout(container)
        outer.setContentsMargins(8, 8, 8, 8)
        outer.setSpacing(12)

        # — Integration time control —
        it_layout = QHBoxLayout()
        it_layout.addWidget(QLabel("Integration Time (s):"))
        self.integrationTimeSpin = QDoubleSpinBox()
        self.integrationTimeSpin.setRange(0.1, 1e4)
        self.integrationTimeSpin.setSingleStep(0.1)
        self.integrationTimeSpin.setValue(1.0)
        it_layout.addWidget(self.integrationTimeSpin)
        outer.addLayout(it_layout)

        # — Save folder selector —
        fld = QHBoxLayout()
        fld.addWidget(QLabel("Save Folder:"))
        self.folderLE = QLineEdit()
        fld.addWidget(self.folderLE, 1)
        b = QPushButton("Browse…")
        b.clicked.connect(self._browse_folder)
        fld.addWidget(b)
        outer.addLayout(fld)

        # — Auxiliary Measurement controls —
        outer.addWidget(QLabel("Aux Measurement:"))
        row = QHBoxLayout()
        self.auxBtn = QPushButton("Measure Aux")
        self.auxBtn.clicked.connect(self.measure_aux)
        row.addWidget(self.auxBtn)

        from PyQt5.QtCore import QTimer
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

        # — PyFai button —
        pyfai_btn = QPushButton("PyFai")
        pyfai_btn.setToolTip("Run pyfai-calib2 in this folder")
        pyfai_btn.clicked.connect(self.run_pyfai)
        outer.addWidget(pyfai_btn)

        # — Real-time controls —
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

        # Controls disabled until hardware is ready
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

    def _toggle_realtime(self, checked: bool):
        if checked:
            self._start_realtime()
            self.rtBtn.setText("Stop RT")
        else:
            self._stop_realtime()
            self.rtBtn.setText("Real-time")

    def _start_realtime(self):
        exposure = float(self.integrationTimeSpin.value())

        # 1) Thread-safe queue for frames
        self._rt_queue = queue.Queue()

        # 2) Set up Matplotlib figure in main thread
        plt.ion()
        self._rt_fig, ax = plt.subplots()
        self._rt_img = ax.imshow(
            np.zeros((256,256)),
            origin='lower',
            interpolation='none'
        )
        plt.show()

        # 3) GUI timer fires frequently to pull newest frame
        self._plot_timer = QTimer(self)
        self._plot_timer.setInterval(50)            # every 50 ms
        self._plot_timer.timeout.connect(self._rt_plot_tick)
        self._plot_timer.start()

        # 4) Worker callback just enqueues every frame
        def callback(frame: np.ndarray):
            # always enqueue; we drop old frames in the GUI thread
            self._rt_queue.put(frame)

        # 5) Kick off the DetectorController loop forever
        self.detector_controller.start_stream(
            callback=callback,
            exposure=exposure,
            interval=0.0,
            frames=1
        )

    def _rt_plot_tick(self):
        """Runs in GUI thread: drain queue and display the very latest frame."""
        frame = None
        while True:
            try:
                frame = self._rt_queue.get_nowait()
            except queue.Empty:
                break

        if frame is not None:
            self._rt_img.set_data(frame)
            self._rt_img.set_clim(frame.min(), frame.max())
            self._rt_fig.canvas.draw_idle()

    def _stop_realtime(self):
        self.detector_controller.stop_stream()
        if hasattr(self, '_plot_timer'):
            self._plot_timer.stop()
            del self._plot_timer
        plt.close(self._rt_fig)
        del self._rt_queue