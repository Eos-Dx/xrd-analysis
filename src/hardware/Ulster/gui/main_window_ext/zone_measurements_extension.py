import json
import os
import time
from copy import copy
from functools import partial
from pathlib import Path

import numpy as np
from PyQt5.QtGui import QColor, QPen
from PyQt5.QtCore import QObject, Qt, QThread, QTimer, pyqtSignal
from PyQt5.QtWidgets import (
    QDockWidget, QDoubleSpinBox, QFileDialog, QHBoxLayout, QLabel, QLineEdit,
    QProgressBar, QPushButton, QSpinBox, QVBoxLayout, QWidget, QListWidget, QListWidgetItem,
    QSpacerItem, QSizePolicy, QGraphicsLineItem, QTabWidget
)
from hardware.Ulster.gui.technical.capture import (
    CaptureWorker,
    move_and_convert_measurement_file,
    validate_folder,
)
from hardware.Ulster.gui.technical.widgets import MeasurementHistoryWidget
from hardware.Ulster.hardware.auxiliary import encode_image_to_base64
from hardware.Ulster.hardware.hardware_control import HardwareController
from xrdanalysis.data_processing.utility_functions import create_mask


from hardware.Ulster.gui.technical.measurement_worker import MeasurementWorker


class ZoneMeasurementsMixin:
    # Signal emitted on hardware initialize (True) or deinitialize (False)
    hardware_state_changed = pyqtSignal(bool)

    def create_zone_measurements_widget(self):
        """
        Creates a dock widget with two tabs:
        1) Measurements (experiment control)
        2) Detector param (masks and PONI for each detector from config)
        """
        self._measurement_threads = []
        self.hardware_initialized = False

        self.zoneMeasurementsDock = QDockWidget("Zone Measurements", self)
        container = QWidget()
        main_layout = QVBoxLayout(container)

        # --- Tab Widget ---
        self.tabs = QTabWidget()
        main_layout.addWidget(self.tabs)

        # ==== Tab 1: Measurements ====
        meas_tab = QWidget()
        meas_layout = QVBoxLayout(meas_tab)
        self.tabs.addTab(meas_tab, "Measurements")

        # -- Measurement controls --
        buttonLayout = QHBoxLayout()
        self.initializeBtn = QPushButton("Initialize Hardware")
        self.initializeBtn.clicked.connect(self.toggle_hardware)
        self.start_btn = QPushButton("Start measurement")
        self.start_btn.clicked.connect(self.start_measurements)
        self.pause_btn = QPushButton("Pause")
        self.pause_btn.clicked.connect(self.pause_measurements)
        self.stop_btn = QPushButton("Stop")
        self.stop_btn.clicked.connect(self.stop_measurements)
        self.start_btn.setEnabled(False)
        self.pause_btn.setEnabled(False)
        self.stop_btn.setEnabled(False)
        buttonLayout.addWidget(self.initializeBtn)
        buttonLayout.addWidget(self.start_btn)
        buttonLayout.addWidget(self.pause_btn)
        buttonLayout.addWidget(self.stop_btn)
        meas_layout.addLayout(buttonLayout)

        # -- Hardware status indicators --
        statusLayout = QHBoxLayout()
        xyLabel = QLabel("XY Stage:")
        self.xyStageIndicator = QLabel()
        self.xyStageIndicator.setFixedSize(20, 20)
        self.xyStageIndicator.setStyleSheet(
            "background-color: gray; border-radius: 10px;"
        )
        statusLayout.addWidget(xyLabel)
        statusLayout.addWidget(self.xyStageIndicator)
        cameraLabel = QLabel("Camera:")
        self.cameraIndicator = QLabel()
        self.cameraIndicator.setFixedSize(20, 20)
        self.cameraIndicator.setStyleSheet(
            "background-color: gray; border-radius: 10px;"
        )
        statusLayout.addWidget(cameraLabel)
        statusLayout.addWidget(self.cameraIndicator)
        self.homeBtn = QPushButton("Home")
        self.homeBtn.clicked.connect(self.home_stage_button_clicked)
        statusLayout.addWidget(self.homeBtn)
        self.loadPosBtn = QPushButton("Load Position")
        self.loadPosBtn.clicked.connect(self.load_position_button_clicked)
        statusLayout.addWidget(self.loadPosBtn)
        meas_layout.addLayout(statusLayout)

        # -- Stage position controls --
        posLayout = QHBoxLayout()
        posLayout.addWidget(QLabel("Stage X (mm):"))
        self.xPosSpin = QDoubleSpinBox()
        self.xPosSpin.setDecimals(3)
        self.xPosSpin.setRange(-1000, 1000)
        self.xPosSpin.setEnabled(False)
        posLayout.addWidget(self.xPosSpin)
        posLayout.addWidget(QLabel("Stage Y (mm):"))
        self.yPosSpin = QDoubleSpinBox()
        self.yPosSpin.setDecimals(3)
        self.yPosSpin.setRange(-1000, 1000)
        self.yPosSpin.setEnabled(False)
        posLayout.addWidget(self.yPosSpin)
        self.gotoBtn = QPushButton("GoTo")
        self.gotoBtn.setEnabled(False)
        self.gotoBtn.clicked.connect(self.goto_stage_position)
        posLayout.addWidget(self.gotoBtn)
        meas_layout.addLayout(posLayout)

        # -- Integration time --
        integrationLayout = QHBoxLayout()
        integrationLabel = QLabel("Integration Time (sec):")
        self.integrationSpinBox = QSpinBox()
        self.integrationSpinBox.setMinimum(1)
        self.integrationSpinBox.setMaximum(600)
        self.integrationSpinBox.setValue(1)
        integrationLayout.addWidget(integrationLabel)
        integrationLayout.addWidget(self.integrationSpinBox)
        meas_layout.addLayout(integrationLayout)

        # -- Repeat count --
        repeatLayout = QHBoxLayout()
        repeatLabel = QLabel("Repeat:")
        self.repeatSpinBox = QSpinBox()
        self.repeatSpinBox.setMinimum(1)
        self.repeatSpinBox.setMaximum(10)
        self.repeatSpinBox.setValue(1)
        repeatLayout.addWidget(repeatLabel)
        repeatLayout.addWidget(self.repeatSpinBox)
        meas_layout.addLayout(repeatLayout)

        # -- Folder selection --
        folderLayout = QHBoxLayout()
        folderLabel = QLabel("Save Folder:")
        self.folderLineEdit = QLineEdit()
        default_folder = (
            self.config.get("default_folder", "")
            if hasattr(self, "config")
            else ""
        )
        self.folderLineEdit.setText(default_folder)
        self.browseBtn = QPushButton("Browse...")
        self.browseBtn.clicked.connect(self.browse_folder)
        folderLayout.addWidget(folderLabel)
        folderLayout.addWidget(self.folderLineEdit)
        folderLayout.addWidget(self.browseBtn)
        meas_layout.addLayout(folderLayout)

        # -- File name --
        fileNameLayout = QHBoxLayout()
        fileNameLabel = QLabel("File Name:")
        self.fileNameLineEdit = QLineEdit()
        fileNameLayout.addWidget(fileNameLabel)
        fileNameLayout.addWidget(self.fileNameLineEdit)
        meas_layout.addLayout(fileNameLayout)

        # -- Additional controls for count and distance --
        additionalLayout = QHBoxLayout()
        self.add_count_btn = QPushButton("Add count")
        self.addCountSpinBox = QSpinBox()
        self.addCountSpinBox.setMinimum(1)
        self.addCountSpinBox.setMaximum(10000)
        self.addCountSpinBox.setValue(60)
        additionalLayout.addWidget(self.add_count_btn)
        additionalLayout.addWidget(self.addCountSpinBox)
        self.add_distance_btn = QPushButton("Add distance")
        self.add_distance_lineedit = QLineEdit("2cm")
        additionalLayout.addWidget(self.add_distance_btn)
        additionalLayout.addWidget(self.add_distance_lineedit)
        additionalLayout.addItem(
            QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)
        )
        meas_layout.addLayout(additionalLayout)
        self.add_distance_btn.clicked.connect(self.handle_add_distance)
        self.add_count_btn.clicked.connect(self.handle_add_count)

        # -- Progress indicator --
        progressLayout = QHBoxLayout()
        self.progressBar = QProgressBar()
        self.progressBar.setMinimum(0)
        self.progressBar.setMaximum(100)
        self.timeRemainingLabel = QLabel("Estimated time: N/A")
        progressLayout.addWidget(self.progressBar)
        progressLayout.addWidget(self.timeRemainingLabel)
        meas_layout.addLayout(progressLayout)

        # ==== Tab 2: Detector param ====
        self.param_tab = QWidget()
        self.param_layout = QVBoxLayout(self.param_tab)
        self.tabs.addTab(self.param_tab, "Detector param")

        container.setLayout(main_layout)
        self.zoneMeasurementsDock.setWidget(container)
        self.addDockWidget(Qt.BottomDockWidgetArea, self.zoneMeasurementsDock)

        self.xyTimer = QTimer(self)
        self.xyTimer.timeout.connect(self.update_xy_pos)
        self.xyTimer.start(10000)

    def populate_detector_param_tab(self):
        # First, clear the param tab layout
        self.clear_detector_param_tab()

        self.detector_aliases = self.hardware_controller.active_detector_aliases

        for alias in self.detector_aliases:
            # Mask widget
            mask_layout = QHBoxLayout()
            mask_layout.addWidget(QLabel(f"{alias} Mask:"))
            mask_lineedit = QLineEdit()
            setattr(self, f"{alias.lower()}_mask_lineedit", mask_lineedit)
            mask_layout.addWidget(mask_lineedit)
            mask_btn = QPushButton("Browse...")
            mask_btn.clicked.connect(partial(self.browse_mask_file, detector=alias))
            mask_layout.addWidget(mask_btn)
            self.param_layout.addLayout(mask_layout)

            # PONI widget
            poni_layout = QHBoxLayout()
            poni_layout.addWidget(QLabel(f"{alias} PONI:"))
            poni_lineedit = QLineEdit()
            setattr(self, f"{alias.lower()}_poni_lineedit", poni_lineedit)
            poni_layout.addWidget(poni_lineedit)
            poni_btn = QPushButton("Browse...")
            poni_btn.clicked.connect(partial(self.browse_poni_file, detector=alias))
            poni_layout.addWidget(poni_btn)
            self.param_layout.addLayout(poni_layout)

    def clear_detector_param_tab(self):
        """Remove all widgets from param_layout."""
        while self.param_layout.count():
            item = self.param_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
            # If the item is a layout, delete its children
            elif item.layout() is not None:
                sub_layout = item.layout()
                while sub_layout.count():
                    sub_item = sub_layout.takeAt(0)
                    sub_widget = sub_item.widget()
                    if sub_widget:
                        sub_widget.deleteLater()
                del sub_layout

    def browse_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Save Folder")
        if folder:
            self.folderLineEdit.setText(folder)

    def browse_mask_file(self, detector):
        mask_file, _ = QFileDialog.getOpenFileName(
            self, f"Select {detector} Mask File", "", "Mask Files (*.mask *.npy *.txt);;All Files (*)"
        )
        if mask_file:
            getattr(self, f"{detector.lower()}_mask_lineedit").setText(mask_file)
            self.load_mask_file(mask_file, detector)

    def browse_poni_file(self, detector):
        poni_file, _ = QFileDialog.getOpenFileName(
            self, f"Select {detector} PONI File", "", "PONI Files (*.poni);;All Files (*)"
        )
        if poni_file:
            getattr(self, f"{detector.lower()}_poni_lineedit").setText(poni_file)
            self.load_poni_file(poni_file, detector)

    def load_default_masks_and_ponis(self):
        """Load masks and poni from config for each detector alias."""
        import os
        from pathlib import Path

        self.masks = {}
        self.ponis = {}
        self.detector_aliases = []

        detectors = self.config.get("detectors", [])
        resource_dir = Path(__file__).resolve().parent.parent.parent / "resources"

        for det_cfg in detectors:
            alias = det_cfg["alias"]
            self.detector_aliases.append(alias)

            # Get detector size from config, fallback to (256, 256)
            det_size = tuple(det_cfg.get("size", {}).values()) if "size" in det_cfg else (256, 256)
            if len(det_size) != 2:
                det_size = (256, 256)

            # Faulty pixels mask (optional)
            mask_file = det_cfg.get("faulty_pixels")
            mask_path = None
            if mask_file:
                mask_path = Path(mask_file)
                if not mask_path.is_absolute():
                    mask_path = resource_dir / mask_path
                try:
                    faulty_pixels = np.load(str(mask_path), allow_pickle=True)
                    # Handle both list/array and possibly None
                    mask = create_mask(faulty_pixels, size=det_size)
                    self.masks[alias] = mask
                except Exception as e:
                    print(f"Failed to load faulty pixels for {alias}: {e}")
                    self.masks[alias] = None
            else:
                self.masks[alias] = None

            # PONI: prefer poni_file, fallback to default_poni, fallback to empty
            poni = ""
            poni_file = det_cfg.get("poni_file")
            if poni_file:
                poni_path = Path(poni_file)
                if not poni_path.is_absolute():
                    poni_path = resource_dir / poni_path
                try:
                    with open(str(poni_path), "r") as f:
                        poni = f.read()
                except Exception as e:
                    print(f"Failed to load poni for {alias}: {e}")
                    poni = ""
            else:
                poni = det_cfg.get("default_poni", "")
            self.ponis[alias] = poni

    def load_mask_file(self, mask_file, detector):
        try:
            data = np.load(mask_file)
            self.masks[detector] = self._create_mask(data)
        except Exception as e:
            print(f"Error loading mask file for {detector}:", e)

    def load_poni_file(self, poni_file, detector):
        try:
            with open(poni_file, 'r') as f:
                self.ponis[detector] = f.read()
        except Exception as e:
            print(f"Error loading PONI file for {detector}:", e)

    def toggle_hardware(self):
        """
        Toggle hardware initialization state. Dynamically (re)builds detector param tab widgets
        for only active detectors after hardware is initialized.
        """
        if not getattr(self, "hardware_initialized", False):
            # --- Initialize hardware using your config-driven HardwareController ---
            from hardware.Ulster.hardware.hardware_control import HardwareController
            self.hardware_controller = HardwareController(self.config)
            res_xystage, res_det = self.hardware_controller.initialize()

            # Use updated controllers from hardware_controller
            self.stage_controller = self.hardware_controller.stage_controller
            self.detector_controller = self.hardware_controller.detectors  # dict: {alias: controller}

            # Update indicators
            self.xyStageIndicator.setStyleSheet(
                "background-color: green; border-radius: 10px;"
                if res_xystage else "background-color: red; border-radius: 10px;"
            )
            self.cameraIndicator.setStyleSheet(
                "background-color: green; border-radius: 10px;"
                if res_det else "background-color: red; border-radius: 10px;"
            )

            ok = res_xystage and res_det
            self.start_btn.setEnabled(ok)
            self.pause_btn.setEnabled(False)
            self.stop_btn.setEnabled(False)
            # Enable X/Y controls if hardware is initialized
            self.xPosSpin.setEnabled(ok)
            self.yPosSpin.setEnabled(ok)
            self.gotoBtn.setEnabled(ok)

            if ok:
                self.populate_detector_param_tab()  # Populate "Detector param" tab now
                self.initializeBtn.setText("Deinitialize Hardware")
                self.hardware_initialized = True
                self.hardware_state_changed.emit(True)
        else:
            # --- Deinitialize hardware and clean up ---
            try:
                self.hardware_controller.deinitialize()
            except Exception as e:
                print(f"Error deinitializing hardware: {e}")

            # Clear the param tab UI
            self.clear_detector_param_tab()

            self.xyStageIndicator.setStyleSheet("background-color: gray; border-radius: 10px;")
            self.cameraIndicator.setStyleSheet("background-color: gray; border-radius: 10px;")
            self.start_btn.setEnabled(False)
            self.pause_btn.setEnabled(False)
            self.stop_btn.setEnabled(False)
            self.xPosSpin.setEnabled(False)
            self.yPosSpin.setEnabled(False)
            self.gotoBtn.setEnabled(False)
            self.initializeBtn.setText("Initialize Hardware")
            self.hardware_initialized = False
            self.hardware_state_changed.emit(False)

    def _create_mask(self, mask_file):
        return create_mask(
            np.load(mask_file),
            (
                self.config["detector_size"]["width"],
                self.config["detector_size"]["height"],
            ),
        )

    def handle_add_count(self):
        # Append the value from addCountSpinBox to fileNameLineEdit.
        current_filename = self.fileNameLineEdit.text()
        appended_value = "_" + str(self.addCountSpinBox.value())
        self.fileNameLineEdit.setText(current_filename + appended_value)

    def handle_add_distance(self):
        # Append the text from add_distance_lineedit to fileNameLineEdit.
        current_filename = self.fileNameLineEdit.text()
        appended_value = "_" + self.add_distance_lineedit.text()
        self.fileNameLineEdit.setText(current_filename + appended_value)

    def home_stage_button_clicked(self):
        """
        Homes the XY stage using the controller.
        """
        if (
            hasattr(self, "stage_controller")
            and self.stage_controller is not None
        ):
            x, y = self.stage_controller.home_stage(home_timeout=10)
            print(f"Home position reached: ({x}, {y})")
            self.xyStageIndicator.setStyleSheet(
                "background-color: green; border-radius: 10px;"
            )
        else:
            print("Stage not initialized.")

    def load_position_button_clicked(self):
        """
        Moves the XY stage to a fixed position when the Load Position button is clicked.
        """
        if (
            hasattr(self, "stage_controller")
            and self.stage_controller is not None
        ):
            new_x, new_y = self.stage_controller.move_stage(
                -15, -6, move_timeout=10
            )
            print(f"Loaded position: ({new_x}, {new_y})")
        else:
            print("Stage not initialized.")

    def start_measurements(self):
        """
        Sorts points by coordinates and then begins the measurement sequence.
        """
        self.measurement_folder = validate_folder(
            self.folderLineEdit.text().strip()
        )
        self.state_path_measurements = (
            Path(self.measurement_folder)
            / f"{self.fileNameLineEdit.text()}_state.json"
        )
        self.manual_save_state()
        self.state_measurements = copy(self.state)
        try:
            with open(self.state_path_measurements, "w") as f:
                self.state_measurements["image_base64"] = (
                    encode_image_to_base64(self.image_view.current_image_path)
                )
                json.dump(self.state_measurements, f, indent=4)
        except Exception as e:
            print(e)

        if self.pointsTable.rowCount() == 0:
            print("No points available for measurement.")
            return

        self.start_btn.setEnabled(False)
        self.pause_btn.setEnabled(True)
        self.stop_btn.setEnabled(True)
        self.stopped = False
        self.paused = False

        # Consolidate and sort measurement points.
        generated_points = self.image_view.points_dict["generated"]["points"]
        user_points = self.image_view.points_dict["user"]["points"]
        all_points = []
        for i, item in enumerate(generated_points):
            center = item.sceneBoundingRect().center()
            # Calculate x_mm and y_mm as needed (adjusting for pixel scale etc.)
            x_mm = (
                self.real_x_pos_mm.value()
                - (center.x() - self.include_center[0])
                / self.pixel_to_mm_ratio
            )
            y_mm = (
                self.real_x_pos_mm.value()
                - (center.y() - self.include_center[1])
                / self.pixel_to_mm_ratio
            )
            all_points.append((i, x_mm, y_mm))
        offset = len(generated_points)
        for j, item in enumerate(user_points):
            center = item.sceneBoundingRect().center()
            x_mm = (
                self.real_x_pos_mm.value()
                - (center.x() - self.include_center[0])
                / self.pixel_to_mm_ratio
            )
            y_mm = (
                self.real_x_pos_mm.value()
                - (center.y() - self.include_center[1])
                / self.pixel_to_mm_ratio
            )
            all_points.append((offset + j, x_mm, y_mm))
        all_points_sorted = sorted(
            all_points, key=lambda tup: (tup[1], tup[2])
        )
        self.sorted_indices = [tup[0] for tup in all_points_sorted]
        self.total_points = len(self.sorted_indices)
        self.current_measurement_sorted_index = 0

        self.progressBar.setMaximum(self.total_points)
        self.progressBar.setValue(0)
        self.integration_time = self.integrationSpinBox.value()
        self.initial_estimate = self.total_points * self.integration_time
        self.measurementStartTime = time.time()
        self.timeRemainingLabel.setText(
            f"Estimated time: {self.initial_estimate:.0f} sec"
        )
        print("Starting measurements in sorted order...")
        self.measure_next_point()

    def measure_next_point(self):
        """
        Moves the stage to each point, performs a measurement,
        and converts the data for both detectors.
        """
        if self.stopped:
            print("Measurement stopped.")
            return
        if self.paused:
            print("Measurement is paused. Waiting for resume.")
            return
        if self.current_measurement_sorted_index >= self.total_points:
            print("All points measured.")
            self.start_btn.setEnabled(True)
            self.pause_btn.setEnabled(False)
            self.stop_btn.setEnabled(False)
            return

        index = self.sorted_indices[self.current_measurement_sorted_index]
        gp = self.image_view.points_dict["generated"]["points"]
        up = self.image_view.points_dict["user"]["points"]
        if index < len(gp):
            self._point_item = gp[index]
            self._zone_item = self.image_view.points_dict["generated"][
                "zones"
            ][index]
        else:
            user_index = index - len(gp)
            self._point_item = up[user_index]
            self._zone_item = self.image_view.points_dict["user"]["zones"][
                user_index
            ]
        self.update_xy_pos()
        center = self._point_item.sceneBoundingRect().center()
        self._x_mm = (
            self.real_x_pos_mm.value()
            - (center.x() - self.include_center[0]) / self.pixel_to_mm_ratio
        )
        self._y_mm = (
            self.real_y_pos_mm.value()
            - (center.y() - self.include_center[1]) / self.pixel_to_mm_ratio
        )

        # Move the stage using the new controller.
        new_x, new_y = self.stage_controller.move_stage(
            self._x_mm, self._y_mm, move_timeout=10
        )

        # Build a common filename base (WITHOUT extension or detector label)
        self._timestamp = time.strftime("%Y%m%d_%H%M%S")
        self._base_name = self.fileNameLineEdit.text().strip()
        txt_filename_base = os.path.join(
            self.measurement_folder,
            f"{self._base_name}_{self._x_mm:.2f}_{self._y_mm:.2f}_{self._timestamp}",
        )

        from PyQt5.QtCore import QThread

        # Launch the dual-capture worker in its own thread
        self.capture_worker = CaptureWorker(
            detector_controller=self.detector_controller,
            integration_time=self.integration_time,
            txt_filename_base=txt_filename_base,  # <--- KEY POINT
        )

        self.capture_thread = QThread()
        self.capture_worker.moveToThread(self.capture_thread)

        # Connect thread start to worker run
        self.capture_thread.started.connect(self.capture_worker.run)

        # Connect signals for cleanup and post-processing
        self.capture_worker.finished.connect(self.on_capture_finished)
        self.capture_worker.finished.connect(self.capture_thread.quit)
        self.capture_worker.finished.connect(self.capture_worker.deleteLater)
        self.capture_thread.finished.connect(self.capture_thread.deleteLater)

        # Start the thread
        self.capture_thread.start()

    def on_capture_finished(self, success: bool, result_files: dict):
        if not success:
            print("[Measurement] capture failed.")
            # handle fail...
            return

        print(f"[Measurement] capture successful: {result_files}")

        # Map aliases to .npy file paths
        npy_files = {}
        for alias, src_file in result_files.items():
            src_path = Path(src_file)
            alias_folder = src_path.parent / alias
            npy_files[alias] = move_and_convert_measurement_file(src_path, alias_folder)

        # Use the current row as before
        current_row = self.sorted_indices[self.current_measurement_sorted_index]

        # Launch a post-processing thread that can handle any number of detectors
        self.spawn_measurement_thread(current_row, npy_files)

        # Visual feedback
        green_brush = QColor(0, 255, 0)
        self._point_item.setBrush(green_brush)
        try:
            if self._zone_item:
                green_zone = QColor(0, 255, 0)
                green_zone.setAlphaF(0.2)
                self._zone_item.setBrush(green_zone)
        except Exception as e:
            print(e)
        QTimer.singleShot(1000, self.measurement_finished)

    def process_measurement_result(self, success, result_files, typ):
        """Handle new measurement files, organize by alias, convert, and update list."""
        if not success:
            print(f"[{typ}] capture failed.")
            self._aux_timer.stop()
            self._aux_status.setText("")
            return {}
        else:
            print(f'[{typ}] capture successful: {result_files}')
            self._aux_timer.stop()
            self._aux_status.setText("Done")

        file_map = {}
        for alias, txt_file in result_files.items():
            txt_path = Path(txt_file)
            det_folder = txt_path.parent / alias
            det_folder.mkdir(parents=True, exist_ok=True)
            new_txt_file = det_folder / txt_path.name
            try:
                txt_path.replace(new_txt_file)
            except Exception as e:
                print(f"[ERROR] Moving file {txt_path} → {new_txt_file}: {e}")
                new_txt_file = txt_path
            try:
                data = np.loadtxt(new_txt_file)
                npy = new_txt_file.with_suffix(".npy")
                np.save(npy, data)
            except Exception as e:
                print(f"Conversion error for {alias}: {e}")
                npy = new_txt_file
            file_map[alias] = str(npy)
            item = QListWidgetItem(f"{alias}: {Path(npy).name}")
            item.setData(Qt.UserRole, str(npy))
            self.auxList.addItem(item)
        return file_map

    def add_measurement_to_table(self, row, results, timestamp=None):
        widget = self.pointsTable.cellWidget(row, 5)
        if not isinstance(widget, MeasurementHistoryWidget):
            widget = MeasurementHistoryWidget(
                masks=self.masks,
                ponis=self.ponis,
                parent=self
            )
            self.pointsTable.setCellWidget(row, 5, widget)
        # Add all results as dict: {alias: {'filename': ..., 'goodness': ...}, ...}
        widget.add_measurement(results, timestamp or getattr(self, "_timestamp", ""))

    def spawn_measurement_thread(self, row, file_map):
        thread = QThread(self)
        worker = MeasurementWorker(
            row=row,
            filenames=file_map,  # now a dict of {alias: file}
            masks=self.masks,
            ponis=self.ponis,
            parent=self,
            hf_cutoff_fraction=0.2,
            columns_to_remove=30
        )
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.measurement_ready.connect(self.add_measurement_to_table)
        worker.measurement_ready.connect(thread.quit)
        worker.measurement_ready.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)
        self._measurement_threads.append((thread, worker))
        thread.start()

    def pause_measurements(self):
        """
        Toggles pause/resume state.
        """
        if not hasattr(self, "paused"):
            self.paused = False
        if not self.paused:
            self.paused = True
            self.pause_btn.setText("Resume")
            print("Measurements paused.")
        else:
            self.paused = False
            self.pause_btn.setText("Pause")
            print("Measurements resumed.")
            self.measure_next_point()

    def stop_measurements(self):
        """
        Stops the measurement process and resets controls.
        """
        self.stopped = True
        self.paused = False
        self.current_measurement_sorted_index = 0
        self.progressBar.setValue(0)
        self.timeRemainingLabel.setText("Measurement stopped.")
        self.start_btn.setEnabled(True)
        self.pause_btn.setText("Pause")
        self.pause_btn.setEnabled(False)
        self.stop_btn.setEnabled(False)
        print("Measurements stopped and reset.")

    def measurement_finished(self):
        """
        Called after one measurement completes; updates progress.
        """
        if self.stopped:
            print("Measurement stopped.")
            return

        self.current_measurement_sorted_index += 1
        self.progressBar.setValue(self.current_measurement_sorted_index)
        elapsed = time.time() - self.measurementStartTime
        if self.current_measurement_sorted_index > 0:
            avg_time = elapsed / self.current_measurement_sorted_index
            remaining = avg_time * (
                self.total_points - self.current_measurement_sorted_index
            )
            percent_complete = (
                self.current_measurement_sorted_index / self.total_points
            ) * 100
            self.timeRemainingLabel.setText(
                f"{percent_complete:.0f}% done, {remaining:.0f} sec remaining"
            )

        if (
            self.current_measurement_sorted_index < self.total_points
            and not self.paused
            and not self.stopped
        ):
            self.measure_next_point()
        else:
            if self.current_measurement_sorted_index >= self.total_points:
                print("All points measured.")
                self.pause_btn.setEnabled(False)
                self.stop_btn.setEnabled(False)
                self.start_btn.setEnabled(True)

    def update_xy_pos(self):
        """
        Updates the XY stage position into the spin boxes.
        """

        if getattr(self, "hardware_initialized", False) and hasattr(
            self, "stage_controller"
        ):
            try:
                x, y = self.stage_controller.get_xy_position()
                self.xPosSpin.setValue(x)
                self.yPosSpin.setValue(y)
            except Exception as e:
                print("Error reading stage pos:", e)
        else:
            x, y = 0, 0
            self.xPosSpin.setValue(0.0)
            self.yPosSpin.setValue(0.0)
            # --- redraw beam cross ---

        # 1) remove old beam items
        old = self.image_view.points_dict.get("beam", [])
        for itm in old:
            self.image_view.scene.removeItem(itm)

        def mm_to_pixels(self, x_mm: float, y_mm: float):
            x = (
                self.real_x_pos_mm.value() - x_mm
            ) * self.pixel_to_mm_ratio + self.include_center[0]
            y = (
                self.real_y_pos_mm.value() - y_mm
            ) * self.pixel_to_mm_ratio + self.include_center[1]
            return x, y

        x, y = mm_to_pixels(self, x, y)

        # 2) if we have valid coords, draw new cross
        if x >= 0 and y >= 0:
            size = 15  # half‐length of each line in scene coordinates
            pen = QPen(Qt.black, 5)

            # horizontal line
            hl = QGraphicsLineItem(x - size, y, x + size, y)
            hl.setPen(pen)
            self.image_view.scene.addItem(hl)

            # vertical line
            vl = QGraphicsLineItem(x, y - size, x, y + size)
            vl.setPen(pen)
            self.image_view.scene.addItem(vl)

            # 3) store for next time
            self.image_view.points_dict["beam"] = [hl, vl]
        else:
            self.image_view.points_dict["beam"] = []

    def goto_stage_position(self):
        """
        Move stage to the X,Y values specified in the spin boxes.
        """
        if hasattr(self, "stage_controller") and getattr(
            self, "hardware_initialized", False
        ):
            x = self.xPosSpin.value()
            y = self.yPosSpin.value()
            try:
                new_x, new_y = self.stage_controller.move_stage(x, y)
                self.xPosSpin.setValue(new_x)
                self.yPosSpin.setValue(new_y)
                self.update_xy_pos()
            except Exception as e:
                print("Error moving stage:", e)
        else:
            print("Stage not initialized; cannot GoTo.")
