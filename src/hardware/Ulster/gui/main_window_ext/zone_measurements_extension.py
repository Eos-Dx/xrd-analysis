import os
import json
import time
import numpy as np
import seaborn as sns
from copy import copy
from pathlib import Path
from functools import partial

from PyQt5.QtWidgets import (
    QDockWidget, QLabel, QSpinBox, QLineEdit, QFileDialog,
    QSpacerItem, QSizePolicy, QProgressBar, QWidget, QHBoxLayout,
    QVBoxLayout, QPushButton, QDialog, QDoubleSpinBox
)
from PyQt5.QtCore import QTimer, Qt, QThread, pyqtSignal
from PyQt5.QtGui import QColor

from xrdanalysis.data_processing.utility_functions import create_mask
from hardware.Ulster.hardware.auxiliary import encode_image_to_base64
from hardware.Ulster.hardware.hardware_control import (
    DetectorController,
    XYStageLibController
)

from hardware.Ulster.gui.technical.capture import CaptureWorker, validate_folder, show_measurement_window

class ZoneMeasurementsMixin:
    # Signal emitted on hardware initialize (True) or deinitialize (False)
    hardware_state_changed = pyqtSignal(bool)

    def create_zone_measurements_widget(self):
        """
        Creates a dock widget for zone measurements with controls and indicators.
        """
        self.hardware_initialized = False

        self.zoneMeasurementsDock = QDockWidget("Zone Measurements", self)
        container = QWidget()
        layout = QVBoxLayout(container)

        # Hardware control buttons.
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
        layout.addLayout(buttonLayout)

        # Hardware status indicators.
        statusLayout = QHBoxLayout()
        xyLabel = QLabel("XY Stage:")
        self.xyStageIndicator = QLabel()
        self.xyStageIndicator.setFixedSize(20, 20)
        self.xyStageIndicator.setStyleSheet("background-color: gray; border-radius: 10px;")
        statusLayout.addWidget(xyLabel)
        statusLayout.addWidget(self.xyStageIndicator)

        cameraLabel = QLabel("Camera:")
        self.cameraIndicator = QLabel()
        self.cameraIndicator.setFixedSize(20, 20)
        self.cameraIndicator.setStyleSheet("background-color: gray; border-radius: 10px;")
        statusLayout.addWidget(cameraLabel)
        statusLayout.addWidget(self.cameraIndicator)

        self.homeBtn = QPushButton("Home")
        self.homeBtn.clicked.connect(self.home_stage_button_clicked)
        statusLayout.addWidget(self.homeBtn)
        self.loadPosBtn = QPushButton("Load Position")
        self.loadPosBtn.clicked.connect(self.load_position_button_clicked)
        statusLayout.addWidget(self.loadPosBtn)

        layout.addLayout(statusLayout)

        # Add X,Y spinboxes and GoTo button
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
        layout.addLayout(posLayout)

        # Integration time.
        integrationLayout = QHBoxLayout()
        integrationLabel = QLabel("Integration Time (sec):")
        self.integrationSpinBox = QSpinBox()
        self.integrationSpinBox.setMinimum(1)
        self.integrationSpinBox.setMaximum(600)
        self.integrationSpinBox.setValue(1)
        integrationLayout.addWidget(integrationLabel)
        integrationLayout.addWidget(self.integrationSpinBox)
        layout.addLayout(integrationLayout)

        # Repeat count.
        repeatLayout = QHBoxLayout()
        repeatLabel = QLabel("Repeat:")
        self.repeatSpinBox = QSpinBox()
        self.repeatSpinBox.setMinimum(1)
        self.repeatSpinBox.setMaximum(10)
        self.repeatSpinBox.setValue(1)
        repeatLayout.addWidget(repeatLabel)
        repeatLayout.addWidget(self.repeatSpinBox)
        layout.addLayout(repeatLayout)

        # Folder selection.
        folderLayout = QHBoxLayout()
        folderLabel = QLabel("Save Folder:")
        self.folderLineEdit = QLineEdit()
        default_folder = self.config.get("default_folder", "") if hasattr(self, "config") else ""
        self.folderLineEdit.setText(default_folder)
        self.browseBtn = QPushButton("Browse...")
        self.browseBtn.clicked.connect(self.browse_folder)
        folderLayout.addWidget(folderLabel)
        folderLayout.addWidget(self.folderLineEdit)
        folderLayout.addWidget(self.browseBtn)
        layout.addLayout(folderLayout)

        # File name.
        fileNameLayout = QHBoxLayout()
        fileNameLabel = QLabel("File Name:")
        self.fileNameLineEdit = QLineEdit()
        fileNameLayout.addWidget(fileNameLabel)
        fileNameLayout.addWidget(self.fileNameLineEdit)
        layout.addLayout(fileNameLayout)

        # Additional controls for count and distance.
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
        additionalLayout.addItem(QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum))
        layout.addLayout(additionalLayout)

        self.add_distance_btn.clicked.connect(self.handle_add_distance)
        self.add_count_btn.clicked.connect(self.handle_add_count)

        # Progress indicator.
        progressLayout = QHBoxLayout()
        self.progressBar = QProgressBar()
        self.progressBar.setMinimum(0)
        self.progressBar.setMaximum(100)
        self.timeRemainingLabel = QLabel("Estimated time: N/A")
        progressLayout.addWidget(self.progressBar)
        progressLayout.addWidget(self.timeRemainingLabel)
        layout.addLayout(progressLayout)

        # Mask file selection.
        maskLayout = QHBoxLayout()
        maskLabel = QLabel("Mask file:")
        self.maskLineEdit = QLineEdit()
        self.maskBrowseBtn = QPushButton("Browse...")
        self.maskBrowseBtn.clicked.connect(self.browse_mask_file)
        maskLayout.addWidget(maskLabel)
        maskLayout.addWidget(self.maskLineEdit)
        maskLayout.addWidget(self.maskBrowseBtn)
        layout.addLayout(maskLayout)

        # PONI file selection.
        poniLayout = QHBoxLayout()
        poniLabel = QLabel("PONI file:")
        self.poniLineEdit = QLineEdit()
        self.poniBrowseBtn = QPushButton("Browse...")
        self.poniBrowseBtn.clicked.connect(self.browse_poni_file)
        poniLayout.addWidget(poniLabel)
        poniLayout.addWidget(self.poniLineEdit)
        poniLayout.addWidget(self.poniBrowseBtn)
        layout.addLayout(poniLayout)

        container.setLayout(layout)
        self.zoneMeasurementsDock.setWidget(container)
        self.addDockWidget(Qt.BottomDockWidgetArea, self.zoneMeasurementsDock)

        self.xyTimer = QTimer(self)
        self.xyTimer.timeout.connect(self.update_xy_pos)
        self.xyTimer.start(10000)

    def toggle_hardware(self):
        """
        Toggle hardware initialization state.
        """
        if not self.hardware_initialized:
            dev_mode = self.config.get("DEV", True)
            serial_str = self.config.get("serial_number_XY", "default_serial")
            self.stage_controller = XYStageLibController(serial_num=serial_str, x_chan=2, y_chan=1, dev=dev_mode)
            res_xystage = self.stage_controller.init_stage()

            self.detector_controller = DetectorController(capture_enabled=True, dev=dev_mode)
            res_det = self.detector_controller.init_detector()

            # Update indicators
            self.xyStageIndicator.setStyleSheet(
                "background-color: green; border-radius: 10px;" if res_xystage else
                "background-color: red; border-radius: 10px;"
            )
            self.cameraIndicator.setStyleSheet(
                "background-color: green; border-radius: 10px;" if res_det else
                "background-color: red; border-radius: 10px;"
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
                self.initializeBtn.setText("Deinitialize Hardware")
                self.hardware_initialized = True
                self.hardware_state_changed.emit(True)
        else:
            # Deinitialize hardware
            try:
                self.stage_controller.deinit()
            except Exception as e:
                print(f"Error deinitializing stage: {e}")
            try:
                self.detector_controller.deinit_detector()
            except Exception as e:
                print(f"Error deinitializing detector: {e}")

            # Reset indicators and controls
            self.xyStageIndicator.setStyleSheet("background-color: gray; border-radius: 10px;")
            self.cameraIndicator.setStyleSheet("background-color: gray; border-radius: 10px;")
            self.start_btn.setEnabled(False)
            self.pause_btn.setEnabled(False)
            self.stop_btn.setEnabled(False)
            # Disable X/Y controls
            self.xPosSpin.setEnabled(False)
            self.yPosSpin.setEnabled(False)
            self.gotoBtn.setEnabled(False)
            self.initializeBtn.setText("Initialize Hardware")
            self.hardware_initialized = False
            self.hardware_state_changed.emit(False)

    def load_default_mask(self):
        faulty_pixels = Path(__file__).resolve().parent.parent.parent / 'resources/faulty_pixels.npy'
        if faulty_pixels.exists():
            self.mask = self._create_mask(faulty_pixels)
        else:
            self.mask = np.array([[]])
            print("Faulty pixels file was not found:", faulty_pixels)

    def browse_poni_file(self):
        """
        Opens a file dialog to select a .poni file,
        populates the line edit, and reads its contents into self.poni.
        """
        poni_file, _ = QFileDialog.getOpenFileName(
            self, "Select PONI File", "", "PONI Files (*.poni);;All Files (*)"
        )
        if poni_file:
            self.poniLineEdit.setText(poni_file)
            try:
                with open(poni_file, 'r') as f:
                    self.poni = f.read()
            except Exception as e:
                print(f"Error reading PONI file: {e}")
                self.poni = ""

    def browse_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Save Folder")
        if folder:
            self.folderLineEdit.setText(folder)

    def browse_mask_file(self):
        """
        Opens a file dialog to select a mask file and loads it.
        """
        mask_file, _ = QFileDialog.getOpenFileName(
            self, "Select Mask File", "", "Mask Files (*.mask *.txt);;All Files (*)"
        )
        if mask_file:
            self.maskLineEdit.setText(mask_file)
            self.load_mask_file(mask_file)

    def _create_mask(self, mask_file):
        return create_mask(np.load(mask_file), (self.config["detector_size"]["width"],
                                                self.config["detector_size"]["height"]))

    def browse_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Save Folder")
        if folder:
            self.folderLineEdit.setText(folder)

    def load_mask_file(self, mask_file):
        """
        Dummy logic to read the selected mask file.
        """
        mask_file = Path(mask_file)
        if mask_file.exists():
            try:
                self.mask = self._create_mask(mask_file)
            except Exception as e:
                print("Error loading mask file:", e)
        else:
            print("Mask file was not found:", mask_file)

    def handle_add_count(self):
        # Append the value from addCountSpinBox to fileNameLineEdit.
        current_filename = self.fileNameLineEdit.text()
        appended_value = '_' + str(self.addCountSpinBox.value())
        self.fileNameLineEdit.setText(current_filename + appended_value)

    def handle_add_distance(self):
        # Append the text from add_distance_lineedit to fileNameLineEdit.
        current_filename = self.fileNameLineEdit.text()
        appended_value = '_' + self.add_distance_lineedit.text()
        self.fileNameLineEdit.setText(current_filename + appended_value)

    def home_stage_button_clicked(self):
        """
        Homes the XY stage using the controller.
        """
        if hasattr(self, 'stage_controller') and self.stage_controller is not None:
            x, y = self.stage_controller.home_stage(home_timeout=10)
            print(f"Home position reached: ({x}, {y})")
            self.xyStageIndicator.setStyleSheet("background-color: green; border-radius: 10px;")
        else:
            print("Stage not initialized.")

    def load_position_button_clicked(self):
        """
        Moves the XY stage to a fixed position when the Load Position button is clicked.
        """
        if hasattr(self, 'stage_controller') and self.stage_controller is not None:
            new_x, new_y = self.stage_controller.move_stage(-10, -15, move_timeout=10)
            print(f"Loaded position: ({new_x}, {new_y})")
        else:
            print("Stage not initialized.")

    def start_measurements(self):
        """
        Sorts points by coordinates and then begins the measurement sequence.
        """
        self.measurement_folder = validate_folder(self.folderLineEdit.text().strip())
        self.state_path_measurements = Path(self.measurement_folder) / f'{self.fileNameLineEdit.text()}_state.json'
        self.manual_save_state()
        self.state_measurements = copy(self.state)
        try:
            with open(self.state_path_measurements, "w") as f:
                self.state_measurements['image_base64'] = encode_image_to_base64(self.image_view.current_image_path)
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
            x_mm = self.real_x_pos_mm.value() - (center.x() - self.include_center[0]) / self.pixel_to_mm_ratio
            y_mm = self.real_x_pos_mm.value() - (center.y() - self.include_center[1]) / self.pixel_to_mm_ratio
            all_points.append((i, x_mm, y_mm))
        offset = len(generated_points)
        for j, item in enumerate(user_points):
            center = item.sceneBoundingRect().center()
            x_mm = self.real_x_pos_mm.value() - (center.x() - self.include_center[0]) / self.pixel_to_mm_ratio
            y_mm = self.real_x_pos_mm.value() - (center.y() - self.include_center[1]) / self.pixel_to_mm_ratio
            all_points.append((offset + j, x_mm, y_mm))
        all_points_sorted = sorted(all_points, key=lambda tup: (tup[1], tup[2]))
        self.sorted_indices = [tup[0] for tup in all_points_sorted]
        self.total_points = len(self.sorted_indices)
        self.current_measurement_sorted_index = 0

        self.progressBar.setMaximum(self.total_points)
        self.progressBar.setValue(0)
        self.integration_time = self.integrationSpinBox.value()
        self.initial_estimate = self.total_points * self.integration_time
        self.measurementStartTime = time.time()
        self.timeRemainingLabel.setText(f"Estimated time: {self.initial_estimate:.0f} sec")
        print("Starting measurements in sorted order...")
        self.measure_next_point()

    def measure_next_point(self):
        """
        Moves the stage to each point, performs a measurement,
        and converts the data.
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
            self._zone_item = self.image_view.points_dict["generated"]["zones"][index]
        else:
            user_index = index - len(gp)
            self._point_item = up[user_index]
            zone_item = self.image_view.points_dict["user"]["zones"][user_index]
        self.update_xy_pos()
        center = self._point_item.sceneBoundingRect().center()
        self._x_mm = self.real_x_pos_mm.value() - (center.x() - self.include_center[0]) / self.pixel_to_mm_ratio
        self._y_mm = self.real_y_pos_mm.value() - (center.y() - self.include_center[1]) / self.pixel_to_mm_ratio

        # Move the stage using the new controller.
        new_x, new_y = self.stage_controller.move_stage(self._x_mm, self._y_mm, move_timeout=10)

        # Build filename.
        self._timestamp = time.strftime("%Y%m%d_%H%M%S")
        self._base_name = self.fileNameLineEdit.text().strip()
        txt_filename = os.path.join(self.measurement_folder,
                                    f"{self._base_name}_{self._x_mm:.2f}_{self._y_mm:.2f}_{self._timestamp}.txt")

        # launch the capture in its own thread:
        self.capture_worker = CaptureWorker(
            detector_controller=self.detector_controller,
            integration_time=self.integration_time,
            txt_filename=txt_filename
        )
        self.capture_worker.finished.connect(self.on_capture_finished)
        self.capture_worker.start()

    def on_capture_finished(self, success: bool, txt_filename: str):
        if success:
            state_path = self.state_path_measurements
            # Build the new entry
            new_meta = {
                Path(txt_filename).name: {
                    'x': self._x_mm,
                    'y': self._y_mm,
                    'base_file': self._base_name,
                    'integration_time': self.integration_time,
                    'distance': self.add_distance_lineedit.text()
                }
            }

            # Get the existing measurements_meta (or an empty dict), update it, and save back
            measurements = self.state_measurements.get('measurements_meta', {})
            measurements.update(new_meta)
            self.state_measurements['measurements_meta'] = measurements

            with open(state_path, 'w') as f:
                json.dump(self.state_measurements, f, indent=4)

            # Convert the captured data.
            try:
                data = np.loadtxt(txt_filename)
                npy_filename = os.path.join(self.measurement_folder,
                                            f"{self._base_name}_{self._x_mm:.2f}_{self._y_mm:.2f}_{self._timestamp}.npy")
                np.save(npy_filename, data)
                print(f"Converted {txt_filename} to {npy_filename}")
            except Exception as e:
                print(f"Error converting file: {e}")
                npy_filename = txt_filename  # Fallback

        self.add_measurement_to_table(self.sorted_indices[self.current_measurement_sorted_index], npy_filename)
        # Visual feedback.
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

    def add_measurement_to_table(self, row, measurement_filename):
        """
        Adds a clickable button to the measurement cell.
        """
        from PyQt5.QtWidgets import QWidget, QHBoxLayout, QPushButton
        widget = self.pointsTable.cellWidget(row, 5)
        if widget is None:
            widget = QWidget()
            layout = QHBoxLayout(widget)
            layout.setContentsMargins(0, 0, 0, 0)
            widget.setLayout(layout)
            self.pointsTable.setCellWidget(row, 5, widget)
        else:
            layout = widget.layout()
        btn = QPushButton(os.path.basename(measurement_filename))
        btn.setStyleSheet("Text-align:left; border: none; color: blue; text-decoration: underline;")
        btn.setCursor(Qt.PointingHandCursor)
        btn.clicked.connect(partial(
            show_measurement_window,
            measurement_filename,
            self.mask,
            getattr(self, "poni", None),  # or just self.poni if you guarantee it exists
            self  # parent window
        ))
        layout.addWidget(btn)

    def pause_measurements(self):
        """
        Toggles pause/resume state.
        """
        if not hasattr(self, 'paused'):
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
            remaining = avg_time * (self.total_points - self.current_measurement_sorted_index)
            percent_complete = (self.current_measurement_sorted_index / self.total_points) * 100
            self.timeRemainingLabel.setText(f"{percent_complete:.0f}% done, {remaining:.0f} sec remaining")

        if self.current_measurement_sorted_index < self.total_points and not self.paused and not self.stopped:
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
        if getattr(self, 'hardware_initialized', False) and hasattr(self, 'stage_controller'):
            try:
                x, y = self.stage_controller.get_xy_position()
                self.xPosSpin.setValue(x)
                self.yPosSpin.setValue(y)
            except Exception as e:
                print("Error reading stage pos:", e)
        else:
            self.xPosSpin.setValue(0.0)
            self.yPosSpin.setValue(0.0)

    def goto_stage_position(self):
        """
        Move stage to the X,Y values specified in the spin boxes.
        """
        if hasattr(self, 'stage_controller') and getattr(self, 'hardware_initialized', False):
            x = self.xPosSpin.value()
            y = self.yPosSpin.value()
            try:
                new_x, new_y = self.stage_controller.move_stage(x, y)
                self.xPosSpin.setValue(new_x)
                self.yPosSpin.setValue(new_y)
            except Exception as e:
                print("Error moving stage:", e)
        else:
            print("Stage not initialized; cannot GoTo.")