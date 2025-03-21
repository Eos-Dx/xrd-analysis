from PyQt5.QtWidgets import QDockWidget, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QSpinBox, QLineEdit, QPushButton, QFileDialog
from PyQt5.QtCore import Qt

class ZoneMeasurementsMixin:
    def createZoneMeasurementsWidget(self):
        """Creates a dock widget for zone measurements.
        This widget (placed at the bottom right) includes:
          - Start, Pause, and Stop buttons.
          - An Integration Time (sec) control.
          - A Repeat control (default 1, min=1, max=10).
          - Controls for selecting a folder where to save and specifying a file name.
        """
        self.zoneMeasurementsDock = QDockWidget("Zone Measurements", self)
        container = QWidget()
        layout = QVBoxLayout(container)

        # Buttons: Start, Pause, Stop
        buttonLayout = QHBoxLayout()
        self.startBtn = QPushButton("Start")
        self.pauseBtn = QPushButton("Pause")
        self.stopBtn = QPushButton("Stop")
        buttonLayout.addWidget(self.startBtn)
        buttonLayout.addWidget(self.pauseBtn)
        buttonLayout.addWidget(self.stopBtn)
        layout.addLayout(buttonLayout)

        # Integration time control
        integrationLayout = QHBoxLayout()
        integrationLabel = QLabel("Integration Time (sec):")
        self.integrationSpinBox = QSpinBox()
        self.integrationSpinBox.setMinimum(1)
        self.integrationSpinBox.setMaximum(60)  # adjust as needed
        self.integrationSpinBox.setValue(1)
        integrationLayout.addWidget(integrationLabel)
        integrationLayout.addWidget(self.integrationSpinBox)
        layout.addLayout(integrationLayout)

        # Repeat control
        repeatLayout = QHBoxLayout()
        repeatLabel = QLabel("Repeat:")
        self.repeatSpinBox = QSpinBox()
        self.repeatSpinBox.setMinimum(1)
        self.repeatSpinBox.setMaximum(10)
        self.repeatSpinBox.setValue(1)
        repeatLayout.addWidget(repeatLabel)
        repeatLayout.addWidget(self.repeatSpinBox)
        layout.addLayout(repeatLayout)

        # Folder selection control
        folderLayout = QHBoxLayout()
        folderLabel = QLabel("Save Folder:")
        self.folderLineEdit = QLineEdit()
        self.browseBtn = QPushButton("Browse...")
        self.browseBtn.clicked.connect(self.browseFolder)
        folderLayout.addWidget(folderLabel)
        folderLayout.addWidget(self.folderLineEdit)
        folderLayout.addWidget(self.browseBtn)
        layout.addLayout(folderLayout)

        # File name control
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
