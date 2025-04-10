import os
import json
import time
import numpy as np
import seaborn as sns
from pathlib import Path
from PyQt5.QtWidgets import (
    QDockWidget, QLabel, QSpinBox, QLineEdit, QFileDialog,
    QSpacerItem, QSizePolicy, QProgressBar, QWidget, QHBoxLayout,
    QVBoxLayout, QPushButton, QDialog
)
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtGui import QColor

from xrdanalysis.data_processing.azimuthal_integration import initialize_azimuthal_integrator_df
from hardware.Ulster.hardware.auxiliary import encode_image_to_base64

# Import the new hardware controllers.
from hardware.Ulster.hardware.hardware_control import DetectorController, XYStageController

class ZoneMeasurementsMixin:

    def createZoneMeasurementsWidget(self):
        """
        Creates a dock widget for zone measurements with controls and indicators.
        """
        self.zoneMeasurementsDock = QDockWidget("Zone Measurements", self)
        container = QWidget()
        layout = QVBoxLayout(container)

        # Hardware control buttons.
        buttonLayout = QHBoxLayout()
        self.initializeBtn = QPushButton("Initialize")
        self.initializeBtn.clicked.connect(self.initializeHardware)
        self.startBtn = QPushButton("Start measurement")
        self.startBtn.clicked.connect(self.startMeasurements)
        self.pauseBtn = QPushButton("Pause")
        self.pauseBtn.clicked.connect(self.pauseMeasurements)
        self.stopBtn = QPushButton("Stop")
        self.stopBtn.clicked.connect(self.stopMeasurements)
        self.startBtn.setEnabled(False)
        self.pauseBtn.setEnabled(False)
        self.stopBtn.setEnabled(False)
        buttonLayout.addWidget(self.initializeBtn)
        buttonLayout.addWidget(self.startBtn)
        buttonLayout.addWidget(self.pauseBtn)
        buttonLayout.addWidget(self.stopBtn)
        layout.addLayout(buttonLayout)

        # Hardware status indicators and extra controls.
        statusLayout = QHBoxLayout()
        xyLabel = QLabel("XY Stage:")
        self.xyStageIndicator = QLabel()
        self.xyStageIndicator.setFixedSize(20, 20)
        self.xyStageIndicator.setStyleSheet("background-color: gray; border-radius: 10px;")
        statusLayout.addWidget(xyLabel)
        statusLayout.addWidget(self.xyStageIndicator)

        self.xyPosLabel = QLabel("Pos: N/A")
        statusLayout.addWidget(self.xyPosLabel)

        cameraLabel = QLabel("Camera:")
        self.cameraIndicator = QLabel()
        self.cameraIndicator.setFixedSize(20, 20)
        self.cameraIndicator.setStyleSheet("background-color: gray; border-radius: 10px;")
        statusLayout.addWidget(cameraLabel)
        statusLayout.addWidget(self.cameraIndicator)

        self.homeBtn = QPushButton("Home")
        self.homeBtn.clicked.connect(self.homeStageButtonClicked)
        statusLayout.addWidget(self.homeBtn)
        self.loadPosBtn = QPushButton("Load Position")
        self.loadPosBtn.clicked.connect(self.loadPositionButtonClicked)
        statusLayout.addWidget(self.loadPosBtn)

        layout.addLayout(statusLayout)

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
        self.browseBtn.clicked.connect(self.browseFolder)
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

        # Progress indicator.
        progressLayout = QHBoxLayout()
        self.progressBar = QProgressBar()
        self.progressBar.setMinimum(0)
        self.progressBar.setMaximum(100)
        self.timeRemainingLabel = QLabel("Estimated time: N/A")
        progressLayout.addWidget(self.progressBar)
        progressLayout.addWidget(self.timeRemainingLabel)
        layout.addLayout(progressLayout)

        container.setLayout(layout)
        self.zoneMeasurementsDock.setWidget(container)
        self.addDockWidget(Qt.BottomDockWidgetArea, self.zoneMeasurementsDock)

        # Assume points_dict is used elsewhere in your app.
        self.image_view.points_dict = {
            "generated": {"points": [], "zones": []},
            "user": {"points": [], "zones": []}
        }

        # Timer to update the XY stage position.
        self.xyTimer = QTimer(self)
        self.xyTimer.timeout.connect(self.updateXYPos)
        self.xyTimer.start(1000)

    def browseFolder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Save Folder")
        if folder:
            self.folderLineEdit.setText(folder)

    def initializeHardware(self):
        """
        Initializes both the XY stage and detector.
        Depending on the self.config['DEV'] flag, this will initialize either dummy
        or real hardware.
        """
        dev_mode = self.config.get("DEV", True)  # Default to DEV mode if key not present

        # Instantiate the controllers.
        # Note: For the stage, pass the serial number and channel settings as needed.
        serial_str = self.config.get("serial_number_XY", "default_serial")
        self.stage_controller = XYStageController(serial_num=serial_str, x_chan=2, y_chan=1, dev=dev_mode)
        self.stage_controller.init_stage()

        self.detector_controller = DetectorController(capture_enabled=True, dev=dev_mode)
        self.detector_controller.init_detector()

        # Update LED indicators based on initialization.
        try:
            # For the stage, if no error occurred, mark green.
            print("XY stage initialized.")
            self.xyStageIndicator.setStyleSheet("background-color: green; border-radius: 10px;")
        except Exception as e:
            print("Error initializing XY stage:", e)
            self.xyStageIndicator.setStyleSheet("background-color: red; border-radius: 10px;")

        try:
            # For the detector.
            print("Pixet camera initialized.")
            self.cameraIndicator.setStyleSheet("background-color: green; border-radius: 10px;")
        except Exception as e:
            print("Error initializing Pixet camera:", e)
            self.cameraIndicator.setStyleSheet("background-color: red; border-radius: 10px;")

        if ("green" in self.xyStageIndicator.styleSheet() and "green" in self.cameraIndicator.styleSheet()):
            self.startBtn.setEnabled(True)
            self.pauseBtn.setEnabled(False)
            self.stopBtn.setEnabled(False)

    def homeStageButtonClicked(self):
        """
        Homes the XY stage using the controller.
        """
        if hasattr(self, 'stage_controller') and self.stage_controller is not None:
            x, y = self.stage_controller.home_stage(home_timeout=10)
            print(f"Home position reached: ({x}, {y})")
            self.xyStageIndicator.setStyleSheet("background-color: green; border-radius: 10px;")
        else:
            print("Stage not initialized.")

    def loadPositionButtonClicked(self):
        """
        Moves the XY stage to a fixed position when the Load Position button is clicked.
        """
        if hasattr(self, 'stage_controller') and self.stage_controller is not None:
            new_x, new_y = self.stage_controller.move_stage(-10, -15, move_timeout=10)
            print(f"Loaded position: ({new_x}, {new_y})")
        else:
            print("Stage not initialized.")

    def startMeasurements(self):
        """
        Sorts points by coordinates and then begins the measurement sequence.
        """
        self.validate_folder()

        self.manualSaveState()
        try:
            with open(Path(self.measurement_folder) / f'{self.fileNameLineEdit.text()}_state.json', "w") as f:
                self.state['image_base64'] = encode_image_to_base64(self.image_view.current_image_path)
                json.dump(self.state, f, indent=4)
        except Exception as e:
            print(e)

        if self.pointsTable.rowCount() == 0:
            print("No points available for measurement.")
            return

        self.startBtn.setEnabled(False)
        self.pauseBtn.setEnabled(True)
        self.stopBtn.setEnabled(True)
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
            y_mm = self.real_y_pos_mm.value() - (center.y() - self.include_center[1]) / self.pixel_to_mm_ratio
            all_points.append((i, x_mm, y_mm))
        offset = len(generated_points)
        for j, item in enumerate(user_points):
            center = item.sceneBoundingRect().center()
            x_mm = self.real_x_pos_mm.value() - (center.x() - self.include_center[0]) / self.pixel_to_mm_ratio
            y_mm = self.real_y_pos_mm.value() - (center.y() - self.include_center[1]) / self.pixel_to_mm_ratio
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
        self.measureNextPoint()

    def measureNextPoint(self):
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
            self.startBtn.setEnabled(True)
            self.pauseBtn.setEnabled(False)
            self.stopBtn.setEnabled(False)
            return

        index = self.sorted_indices[self.current_measurement_sorted_index]
        gp = self.image_view.points_dict["generated"]["points"]
        up = self.image_view.points_dict["user"]["points"]
        if index < len(gp):
            point_item = gp[index]
            zone_item = self.image_view.points_dict["generated"]["zones"][index]
        else:
            user_index = index - len(gp)
            point_item = up[user_index]
            zone_item = self.image_view.points_dict["user"]["zones"][user_index]

        center = point_item.sceneBoundingRect().center()
        x_mm = self.real_x_pos_mm.value() - (center.x() - self.include_center[0]) / self.pixel_to_mm_ratio
        y_mm = self.real_x_pos_mm.value() - (center.y() - self.include_center[1]) / self.pixel_to_mm_ratio

        # Move the stage using the new controller.
        new_x, new_y = self.stage_controller.move_stage(x_mm, y_mm, move_timeout=10)

        # Build filename.
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        base_name = self.fileNameLineEdit.text().strip()
        txt_filename = os.path.join(self.measurement_folder, f"{base_name}_{x_mm:.2f}_{y_mm:.2f}_{timestamp}.txt")

        # Capture measurement using the detector controller.
        self.detector_controller.capture_point(1, self.integration_time, txt_filename)

        # Convert the captured data.
        try:
            data = np.loadtxt(txt_filename)
            npy_filename = os.path.join(self.measurement_folder, f"{base_name}_{x_mm:.2f}_{y_mm:.2f}_{timestamp}.npy")
            np.save(npy_filename, data)
            print(f"Converted {txt_filename} to {npy_filename}")
        except Exception as e:
            print(f"Error converting file: {e}")
            npy_filename = txt_filename  # Fallback

        self.addMeasurementToTable(self.sorted_indices[self.current_measurement_sorted_index], npy_filename)

        # Visual feedback.
        green_brush = QColor(0, 255, 0)
        point_item.setBrush(green_brush)
        if zone_item:
            green_zone = QColor(0, 255, 0)
            green_zone.setAlphaF(0.2)
            zone_item.setBrush(green_zone)

        QTimer.singleShot(1000, self.measurementFinished)

    def addMeasurementToTable(self, row, measurement_filename):
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
        btn.clicked.connect(lambda: self.showMeasurement(measurement_filename))
        layout.addWidget(btn)

    def showMeasurement(self, measurement_filename):
        """
        Opens a window that shows the raw 2D image and its azimuthal integration.
        Uses Seaborn to render the heatmap and lineplot.
        """
        data = np.load(measurement_filename)

        # Calibration parameters.
        pixel_size = 55e-6
        max_idx = np.unravel_index(np.argmax(data), data.shape)
        center_row, center_column = max_idx
        wavelength = 1.54
        sample_distance_mm = 100.0

        # Initialize integrator.
        ai = initialize_azimuthal_integrator_df(
            pixel_size,
            center_column,
            center_row,
            wavelength,
            sample_distance_mm
        )

        npt = 100  # Number of integration points.
        try:
            result = ai.integrate1d(data, npt, unit="q_nm^-1", error_model="azimuthal")
            radial = result.radial
            intensity = result.intensity
        except Exception as e:
            print("Error integrating data:", e)
            return

        dialog = QDialog(self)
        dialog.setWindowTitle(f"Azimuthal Integration: {os.path.basename(measurement_filename)}")
        layout = QHBoxLayout(dialog)

        # Left plot: raw 2D image using seaborn heatmap.
        fig1 = Figure(figsize=(5, 5))
        canvas1 = FigureCanvas(fig1)
        ax1 = fig1.add_subplot(111)
        sns.heatmap(data, robust=True, square=True, cmap="viridis", ax=ax1)
        ax1.set_title("2D Image")
        # seaborn.heatmap by default displays a colorbar.

        # Right plot: integration result with y-axis on log scale using seaborn.
        fig2 = Figure(figsize=(5, 5))
        canvas2 = FigureCanvas(fig2)
        ax2 = fig2.add_subplot(111)
        sns.lineplot(x=radial, y=intensity, marker='o', ax=ax2)
        ax2.set_title("Azimuthal Integration")
        ax2.set_xlabel("q (nm^-1)")
        ax2.set_ylabel("Intensity")
        ax2.set_yscale("log")  # Set y-axis to log scale

        layout.addWidget(canvas1)
        layout.addWidget(canvas2)
        dialog.resize(1000, 500)

        if not hasattr(self, "_open_measurement_windows"):
            self._open_measurement_windows = []
        self._open_measurement_windows.append(dialog)
        dialog.finished.connect(lambda _: self._open_measurement_windows.remove(dialog))
        dialog.show()

    def pauseMeasurements(self):
        """
        Toggles pause/resume state.
        """
        if not hasattr(self, 'paused'):
            self.paused = False
        if not self.paused:
            self.paused = True
            self.pauseBtn.setText("Resume")
            print("Measurements paused.")
        else:
            self.paused = False
            self.pauseBtn.setText("Pause")
            print("Measurements resumed.")
            self.measureNextPoint()

    def stopMeasurements(self):
        """
        Stops the measurement process and resets controls.
        """
        self.stopped = True
        self.paused = False
        self.current_measurement_sorted_index = 0
        self.progressBar.setValue(0)
        self.timeRemainingLabel.setText("Measurement stopped.")
        self.startBtn.setEnabled(True)
        self.pauseBtn.setText("Pause")
        self.pauseBtn.setEnabled(False)
        self.stopBtn.setEnabled(False)
        print("Measurements stopped and reset.")

    def measurementFinished(self):
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
            self.measureNextPoint()
        else:
            if self.current_measurement_sorted_index >= self.total_points:
                print("All points measured.")
                self.pauseBtn.setEnabled(False)
                self.stopBtn.setEnabled(False)
                self.startBtn.setEnabled(True)

    def validate_folder(self):
        """
        Validates the selected folder.
        """
        self.measurement_folder = self.folderLineEdit.text().strip()
        if not self.measurement_folder:
            self.measurement_folder = os.getcwd()
        if not os.path.exists(self.measurement_folder):
            try:
                os.makedirs(self.measurement_folder, exist_ok=True)
            except Exception as e:
                print(f"Error creating folder {self.measurement_folder}: {e}. Using current directory.")
                self.measurement_folder = os.getcwd()
        if not os.access(self.measurement_folder, os.W_OK):
            print(f"Folder {self.measurement_folder} is not writable. Using current directory.")
            self.measurement_folder = os.getcwd()

    def updateXYPos(self):
        """
        Updates the XY stage position.
        """
        if hasattr(self, 'stage_controller') and self.stage_controller is not None:
            try:
                pos = self.stage_controller.get_xy_position()
                self.xyPosLabel.setText(f"Pos: ({pos[0]:.2f}, {pos[1]:.2f})")
            except Exception as e:
                self.xyPosLabel.setText("Pos: Error")
        else:
            self.xyPosLabel.setText("Pos: N/A")
