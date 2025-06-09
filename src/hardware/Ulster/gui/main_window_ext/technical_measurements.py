import os
import time
import numpy as np
import subprocess

from PyQt5.QtWidgets import (
    QDockWidget, QWidget, QHBoxLayout, QVBoxLayout,
    QLabel, QLineEdit, QPushButton, QFileDialog,
    QListWidget, QListWidgetItem, QDoubleSpinBox
)
from PyQt5.QtCore import Qt

from hardware.Ulster.gui.technical.capture import CaptureWorker, validate_folder
from hardware.Ulster.gui.main_window_ext.zone_measurements_extension import ZoneMeasurementsMixin
from hardware.Ulster.gui.technical.capture import show_measurement_window


class TechnicalMeasurementsMixin(ZoneMeasurementsMixin):

    def create_measurements_panel(self):
        # Initialize counters
        self.empty_counter = 0
        self.background_counter = 0
        self.calibrant_counter = 0

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

        # Measurement types
        for typ in ("Empty", "Background", "Calibrant"):
            outer.addWidget(QLabel(f"{typ} Measurements:"))
            row = QHBoxLayout()
            btn = QPushButton(f"Measure {typ}")
            btn.clicked.connect(getattr(self, f"measure_{typ.lower()}"))
            setattr(self, f"{typ.lower()}Btn", btn)
            row.addWidget(btn)

            le = QLineEdit()
            le.setPlaceholderText(f"Name for {typ}")
            setattr(self, f"{typ.lower()}NameLE", le)
            row.addWidget(le, 1)
            outer.addLayout(row)

            lst = QListWidget()
            lst.itemActivated.connect(self.open_measurement)
            setattr(self, f"{typ.lower()}List", lst)
            outer.addWidget(lst)

            # **Only** for Calibrant: add the PyFai button
            if typ == "Calibrant":
                pyfai_btn = QPushButton("PyFai")
                pyfai_btn.setToolTip("Run pyfai-calib2 in this folder")
                pyfai_btn.clicked.connect(self.run_pyfai)
                outer.addWidget(pyfai_btn)

        self.measDock.setWidget(container)
        self.measDock.setWidget(container)
        # Wrap in a scroll area so the contents can scroll if they exceed the dock size
        from PyQt5.QtWidgets import QScrollArea
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(container)
        self.measDock.setWidget(scroll)
        self.addDockWidget(Qt.LeftDockWidgetArea, self.measDock)

        # Initially disable controls until hardware is initialized
        self.enable_measurement_controls(False)

        # Subscribe to hardware init/deinit events
        # ZoneMeasurementsMixin emits hardware_state_changed(bool)
        self.hardware_state_changed.connect(self.enable_measurement_controls)

    def run_pyfai(self):
        env = self.config.get("conda")
        if not env:
            print("❌ No conda env set in self.config['conda']")
            return

        folder = validate_folder(self.folderLE.text())

        if os.name == "nt":
            # Build a single string that start.exe can launch
            cmd = (
                f'CALL conda activate {env} '
                f'&& cd /d "{folder}" '
                f'&& pyfai-calib2'
            )
            start_cmd = f'start cmd /K "{cmd}"'
            try:
                # shell=True is required so 'start' is recognized
                subprocess.Popen(start_cmd, shell=True)
                print("▶️ Launched PyFai in new cmd window.")
            except Exception as e:
                print("❌ Failed to launch PyFai on Windows:", e)

        else:
            # Unix branch unchanged...
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

    def enable_measurement_controls(self, enable: bool):
        widgets = [
            self.integrationTimeSpin, self.folderLE,
            self.emptyBtn, self.emptyNameLE, self.emptyList,
            self.backgroundBtn, self.backgroundNameLE, self.backgroundList,
            self.calibrantBtn, self.calibrantNameLE, self.calibrantList
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
        txt = os.path.join(folder, f"{base_with_count}_{ts}.txt")

        worker = CaptureWorker(
            detector_controller=self.detector_controller,
            integration_time=self.integrationTimeSpin.value(),
            txt_filename=txt
        )

        # Keep it alive until it finishes:
        if not hasattr(self, "_capture_workers"):
            self._capture_workers = []
        self._capture_workers.append(worker)

        # Clean up when done:
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
            return
        else:
            print('[{typ}] capture successful:', txt_file)
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

    def measure_empty(self):
        self._start_capture("Empty")

    def measure_background(self):
        self._start_capture("Background")

    def measure_calibrant(self):
        self._start_capture("Calibrant")

    def open_measurement(self, item: QListWidgetItem):
        show_measurement_window(
            item.data(Qt.UserRole),
            self.mask,
            getattr(self, "poni", None),
            self
        )

    # Override to subscribe automatically
    def initialize_hardware(self):
        super().initialize_hardware()
        # No need to manually toggle controls here;
        # subscription to hardware_state_changed handles it.
