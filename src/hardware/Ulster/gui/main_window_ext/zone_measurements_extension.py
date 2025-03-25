import random
import math
from PyQt5.QtWidgets import (
    QDockWidget, QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QSpinBox, QLineEdit, QPushButton, QFileDialog, QTableWidget, QTableWidgetItem,
    QGraphicsEllipseItem, QComboBox
)
from PyQt5.QtCore import Qt, QEvent, QPointF, QRectF, QTimer
from PyQt5.QtGui import QColor, QPen, QTransform

# --- ZoneMeasurementsMixin with hardware control and measurement sequence ---
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
          - LED indicators for the XY stage and Camera.
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
        self.stopBtn = QPushButton("Stop")
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

        # Enable the Start, Pause, and Stop buttons only if both hardware components are active.
        if ("green" in self.xyStageIndicator.styleSheet() and 
            "green" in self.cameraIndicator.styleSheet()):
            self.startBtn.setEnabled(True)
            self.pauseBtn.setEnabled(True)
            self.stopBtn.setEnabled(True)

    def startMeasurements(self):
        """
        When the Start measurement button is clicked, if there are points in the table,
        the application iterates through the points one by one.
        For each point, its marker (and associated zone, if present) changes to green (with 20% opacity for the zone)
        to indicate that measurement is in progress.
        """
        if self.pointsTable.rowCount() == 0:
            print("No points available for measurement.")
            return

        # Disable the measurement control buttons to prevent interference.
        self.startBtn.setEnabled(False)
        self.pauseBtn.setEnabled(False)
        self.stopBtn.setEnabled(False)
        self.current_measurement_index = 0
        print("Starting measurements...")
        self.measureNextPoint()

    def measureNextPoint(self):
        total_points = self.pointsTable.rowCount()
        if self.current_measurement_index >= total_points:
            print("All points measured.")
            # Optionally re-enable buttons after all measurements.
            return

        row = self.current_measurement_index
        # Determine if the point is a generated point or user-defined.
        generated_count = len(self.image_view.generated_points) if hasattr(self.image_view, 'generated_points') else 0
        if row < generated_count:
            point_item = self.image_view.generated_points[row]
            zone_item = self.image_view.generated_cyan[row] if row < len(self.image_view.generated_cyan) else None
        else:
            user_index = row - generated_count
            point_item = self.user_defined_points[user_index] if hasattr(self, "user_defined_points") and user_index < len(self.user_defined_points) else None
            zone_item = None

        # Change the color to green to indicate that the point is being measured.
        green_brush = QColor(0, 255, 0)
        if point_item:
            point_item.setBrush(green_brush)
        if zone_item:
            green_zone = QColor(0, 255, 0)
            green_zone.setAlphaF(0.2)
            zone_item.setBrush(green_zone)

        # Simulate the measurement process with a delay of 1 second.
        # Replace this with the actual measurement call as needed.
        QTimer.singleShot(1000, self.measurementFinished)

    def measurementFinished(self):
        """
        Called after a simulated measurement delay.
        Proceeds to the next point.
        """
        self.current_measurement_index += 1
        self.measureNextPoint()