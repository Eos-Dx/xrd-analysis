import random
import math
import time
from PyQt5.QtWidgets import (
    QDockWidget, QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QSpinBox, QLineEdit, QPushButton, QFileDialog, QTableWidget, QTableWidgetItem,
    QGraphicsEllipseItem, QComboBox, QSpacerItem, QSizePolicy, QProgressBar
)
from PyQt5.QtCore import Qt, QEvent, QPointF, QRectF, QTimer
from PyQt5.QtGui import QColor, QPen, QTransform

# --- ZoneMeasurementsMixin with hardware control, measurement sequence, progress visualization,
#      and sorted measurement order along growing X (mm) and Y (mm) coordinates ---
class ZoneMeasurementsMixin:
    def createZoneMeasurementsWidget(self):
        """
        Creates a dock widget for zone measurements.
        This widget (placed at the bottom right) includes:
          - An Initialize button (to set up hardware),
          - Start, Pause, and Stop buttons,
          - An Integration Time (sec) control,
          - A Repeat control,
          - Folder and File Name controls,
          - LED indicators for the XY stage and Camera,
          - A progress indicator showing percentage complete and estimated time remaining.
        """
        self.zoneMeasurementsDock = QDockWidget("Zone Measurements", self)
        container = QWidget()
        layout = QVBoxLayout(container)

        # Hardware control buttons: Initialize, Start, Pause, Stop.
        buttonLayout = QHBoxLayout()
        self.initializeBtn = QPushButton("Initialize")
        self.initializeBtn.clicked.connect(self.initializeHardware)
        self.startBtn = QPushButton("Start measurement")
        self.startBtn.clicked.connect(self.startMeasurements)
        self.pauseBtn = QPushButton("Pause")
        self.pauseBtn.clicked.connect(self.pauseMeasurements)
        self.stopBtn = QPushButton("Stop")
        self.stopBtn.clicked.connect(self.stopMeasurements)
        # Initially, these buttons are disabled until hardware is initialized.
        self.startBtn.setEnabled(False)
        self.pauseBtn.setEnabled(False)
        self.stopBtn.setEnabled(False)
        buttonLayout.addWidget(self.initializeBtn)
        buttonLayout.addWidget(self.startBtn)
        buttonLayout.addWidget(self.pauseBtn)
        buttonLayout.addWidget(self.stopBtn)
        layout.addLayout(buttonLayout)

        # Hardware status indicators for XY stage and Camera.
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
        # Add a spacer item to push all widgets to the left.
        spacer = QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)
        statusLayout.addItem(spacer)
        layout.addLayout(statusLayout)

        # Integration time control.
        integrationLayout = QHBoxLayout()
        integrationLabel = QLabel("Integration Time (sec):")
        self.integrationSpinBox = QSpinBox()
        self.integrationSpinBox.setMinimum(1)
        self.integrationSpinBox.setMaximum(60)
        self.integrationSpinBox.setValue(1)
        integrationLayout.addWidget(integrationLabel)
        integrationLayout.addWidget(self.integrationSpinBox)
        layout.addLayout(integrationLayout)

        # Repeat control.
        repeatLayout = QHBoxLayout()
        repeatLabel = QLabel("Repeat:")
        self.repeatSpinBox = QSpinBox()
        self.repeatSpinBox.setMinimum(1)
        self.repeatSpinBox.setMaximum(10)
        self.repeatSpinBox.setValue(1)
        repeatLayout.addWidget(repeatLabel)
        repeatLayout.addWidget(self.repeatSpinBox)
        layout.addLayout(repeatLayout)

        # Folder selection control.
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

        # File name control.
        fileNameLayout = QHBoxLayout()
        fileNameLabel = QLabel("File Name:")
        self.fileNameLineEdit = QLineEdit()
        fileNameLayout.addWidget(fileNameLabel)
        fileNameLayout.addWidget(self.fileNameLineEdit)
        layout.addLayout(fileNameLayout)

        # Progress indicator layout.
        progressLayout = QHBoxLayout()
        self.progressBar = QProgressBar()
        self.progressBar.setMinimum(0)
        self.progressBar.setMaximum(100)  # Will be reset when measurements start.
        self.timeRemainingLabel = QLabel("Estimated time: N/A")
        progressLayout.addWidget(self.progressBar)
        progressLayout.addWidget(self.timeRemainingLabel)
        layout.addLayout(progressLayout)

        container.setLayout(layout)
        self.zoneMeasurementsDock.setWidget(container)
        self.addDockWidget(Qt.BottomDockWidgetArea, self.zoneMeasurementsDock)

    def browseFolder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Save Folder")
        if folder:
            self.folderLineEdit.setText(folder)

    def initializeHardware(self):
        """
        Initializes the XY stage and Pixet camera.
        Updates LED indicators and enables the control buttons if both hardware components are active.
        """
        try:
            from hardware.Ulster.hardware import hardware_control as hc
        except Exception as e:
            print("Error importing hardware control module:", e)
            self.xyStageIndicator.setStyleSheet("background-color: red; border-radius: 10px;")
            self.cameraIndicator.setStyleSheet("background-color: red; border-radius: 10px;")
            return

        # Initialize XY stage.
        try:
            self.xystage_lib = hc.init_stage(sim_en=True, serial_num=1, x_chan=1, y_chan=2)
            print("XY stage initialized.")
            self.xyStageIndicator.setStyleSheet("background-color: green; border-radius: 10px;")
        except Exception as e:
            print("Error initializing XY stage:", e)
            self.xyStageIndicator.setStyleSheet("background-color: red; border-radius: 10px;")

        # Initialize Pixet camera.
        try:
            self.pixet, self.detector = hc.init_detector(capture_enabled=True)
            print("Pixet camera initialized.")
            self.cameraIndicator.setStyleSheet("background-color: green; border-radius: 10px;")
        except Exception as e:
            print("Error initializing Pixet camera:", e)
            self.cameraIndicator.setStyleSheet("background-color: red; border-radius: 10px;")

        # Enable the Start button only if both hardware components are active.
        if ("green" in self.xyStageIndicator.styleSheet() and
            "green" in self.cameraIndicator.styleSheet()):
            self.startBtn.setEnabled(True)
            self.pauseBtn.setEnabled(False)
            self.stopBtn.setEnabled(False)

    def startMeasurements(self):
        """
        When the Start measurement button is clicked, if there are points in the table,
        the application first sorts the points in increasing order of X (mm) and Y (mm) and then
        iterates through them one by one. For each point, its marker (and associated zone, if present)
        changes to green (with 20% opacity for the zone) to indicate that measurement is in progress.
        Also initializes progress visualization based on the number of points and integration time.
        """
        if self.pointsTable.rowCount() == 0:
            print("No points available for measurement.")
            return

        # Disable Start button; enable Pause and Stop.
        self.startBtn.setEnabled(False)
        self.pauseBtn.setEnabled(True)
        self.stopBtn.setEnabled(True)
        self.stopped = False
        self.paused = False
        self.pauseBtn.setText("Pause")

        # --- Create a sorted order of measurement indices ---
        # Assumes that generated points are stored in self.image_view.generated_points
        # and user-defined points in self.user_defined_points.
        all_points = []
        # Process generated points.
        if hasattr(self.image_view, 'generated_points'):
            for i, item in enumerate(self.image_view.generated_points):
                center = item.sceneBoundingRect().center()
                x_mm = center.x() / self.pixel_to_mm_ratio if self.pixel_to_mm_ratio else center.x()
                y_mm = center.y() / self.pixel_to_mm_ratio if self.pixel_to_mm_ratio else center.y()
                all_points.append((i, x_mm, y_mm))
        # Process user-defined points.
        if hasattr(self, 'user_defined_points'):
            offset = len(self.image_view.generated_points) if hasattr(self.image_view, 'generated_points') else 0
            for j, item in enumerate(self.user_defined_points):
                center = item.sceneBoundingRect().center()
                x_mm = center.x() / self.pixel_to_mm_ratio if self.pixel_to_mm_ratio else center.x()
                y_mm = center.y() / self.pixel_to_mm_ratio if self.pixel_to_mm_ratio else center.y()
                all_points.append((offset + j, x_mm, y_mm))
        # Sort points first by increasing X (mm), then by increasing Y (mm)
        all_points_sorted = sorted(all_points, key=lambda tup: (tup[1], tup[2]))
        # Save sorted indices order.
        self.sorted_indices = [tup[0] for tup in all_points_sorted]
        self.total_points = len(self.sorted_indices)
        self.current_measurement_sorted_index = 0

        # Set up progress visualization.
        self.progressBar.setMaximum(self.total_points)
        self.progressBar.setValue(0)
        self.integration_time = self.integrationSpinBox.value()
        self.initial_estimate = self.total_points * self.integration_time
        self.measurementStartTime = time.time()
        self.timeRemainingLabel.setText(f"Estimated time: {self.initial_estimate:.0f} sec")
        print("Starting measurements in sorted order...")
        self.measureNextPoint()

    def pauseMeasurements(self):
        """
        Toggles the pause/resume state.
        When paused, the measurement loop is halted and the button text changes to "Resume".
        When resumed, measurement continues.
        """
        if not hasattr(self, 'paused'):
            self.paused = False
        if not self.paused:
            self.paused = True
            self.pauseBtn.setEnabled(True)
            self.pauseBtn.setText("Resume")
            print("Measurements paused.")
        else:
            self.paused = False
            self.pauseBtn.setText("Pause")
            print("Measurements resumed.")
            # Resume measurements.
            self.measureNextPoint()

    def stopMeasurements(self):
        """
        Stops the measurement process.
        Resets the measurement state and progress visualization.
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

    def measureNextPoint(self):
        """
        Proceeds to measure the next point unless the process is paused or stopped.
        For each point, changes its color to green to indicate measurement in progress,
        then simulates a measurement delay.
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

        # Get the next measurement index from the sorted order.
        index = self.sorted_indices[self.current_measurement_sorted_index]
        generated_count = len(self.image_view.generated_points) if hasattr(self.image_view, 'generated_points') else 0
        if index < generated_count:
            point_item = self.image_view.generated_points[index]
            zone_item = self.image_view.generated_cyan[index] if index < len(self.image_view.generated_cyan) else None
        else:
            user_index = index - generated_count
            point_item = self.user_defined_points[user_index] if hasattr(self, "user_defined_points") and user_index < len(self.user_defined_points) else None
            zone_item = None

        # Change the marker color to green to indicate measurement in progress.
        green_brush = QColor(0, 255, 0)
        if point_item:
            point_item.setBrush(green_brush)
        if zone_item:
            green_zone = QColor(0, 255, 0)
            green_zone.setAlphaF(0.2)
            zone_item.setBrush(green_zone)

        # Simulate the measurement process with a delay of 1 second.
        QTimer.singleShot(1000, self.measurementFinished)

    def measurementFinished(self):
        """
        Called after a simulated measurement delay.
        Updates progress visualization and proceeds to the next point if not paused or stopped.
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
            self.startBtn.setEnabled(True)
            self.pauseBtn.setEnabled(False)
            self.stopBtn.setEnabled(False)
