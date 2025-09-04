# zone_measurements/logic/ui_mixin.py

from PyQt5.QtWidgets import (
    QCheckBox,
    QDoubleSpinBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)


class ZoneMeasurementsUIMixin:
    def _append_measurement_log(self, msg: str):
        try:
            import time

            ts = time.strftime("%H:%M:%S")
            line = f"{ts} | {msg}"
            if hasattr(self, "measurementLog") and self.measurementLog is not None:
                if (
                    getattr(self, "logCheckBox", None) is None
                    or self.logCheckBox.isChecked()
                ):
                    self.measurementLog.appendPlainText(line)
        except Exception:
            pass

    def create_zone_measurements_widget(self):
        """
        Builds the Measurements tab and all its controls.
        Should add the tab to your tab widget.
        """
        self._measurement_threads = []
        self.hardware_initialized = False

        meas_tab = QWidget()
        meas_layout = QVBoxLayout(meas_tab)
        self.tabs.addTab(meas_tab, "Measurements")

        # --- Measurement controls ---
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

        # --- Hardware status indicators ---
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

        # --- Stage position controls ---
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

        # --- Integration + Attenuation ---
        integrationLayout = QHBoxLayout()
        integrationLabel = QLabel("Integration Time (sec):")
        self.integrationSpinBox = QSpinBox()
        self.integrationSpinBox.setMinimum(1)
        self.integrationSpinBox.setMaximum(600)
        self.integrationSpinBox.setValue(1)
        integrationLayout.addWidget(integrationLabel)
        integrationLayout.addWidget(self.integrationSpinBox)

        # Attenuation controls (checkbox + frames + short time) on the same line
        self.attenuationCheckBox = QCheckBox("Attenuation")
        # Defaults from config
        atten_cfg = (
            self.config.get("attenuation", {}) if hasattr(self, "config") else {}
        )
        enabled_default = bool(atten_cfg.get("enabled_default", False))
        self.attenuationCheckBox.setChecked(enabled_default)
        integrationLayout.addWidget(self.attenuationCheckBox)

        integrationLayout.addWidget(QLabel("Frames:"))
        self.attenFramesSpin = QSpinBox()
        self.attenFramesSpin.setRange(1, 100000)
        self.attenFramesSpin.setValue(int(atten_cfg.get("frames", 100)))
        integrationLayout.addWidget(self.attenFramesSpin)

        integrationLayout.addWidget(QLabel("Short t (s):"))
        self.attenTimeSpin = QDoubleSpinBox()
        self.attenTimeSpin.setDecimals(6)
        self.attenTimeSpin.setRange(0.000001, 10.0)
        self.attenTimeSpin.setValue(float(atten_cfg.get("integration_time_s", 0.00005)))
        integrationLayout.addWidget(self.attenTimeSpin)

        meas_layout.addLayout(integrationLayout)

        # --- Folder selection ---
        folderLayout = QHBoxLayout()
        folderLabel = QLabel("Save Folder:")
        self.folderLineEdit = QLineEdit()
        default_folder = (
            self.config.get("default_folder", "") if hasattr(self, "config") else ""
        )
        self.folderLineEdit.setText(default_folder)
        self.browseBtn = QPushButton("Browse...")
        self.browseBtn.clicked.connect(self.browse_folder)
        folderLayout.addWidget(folderLabel)
        folderLayout.addWidget(self.folderLineEdit)
        folderLayout.addWidget(self.browseBtn)
        meas_layout.addLayout(folderLayout)

        # --- File name ---
        fileNameLayout = QHBoxLayout()
        fileNameLabel = QLabel("File Name:")
        self.fileNameLineEdit = QLineEdit()
        fileNameLayout.addWidget(fileNameLabel)
        fileNameLayout.addWidget(self.fileNameLineEdit)
        meas_layout.addLayout(fileNameLayout)

        # --- Additional controls for count and distance ---
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
        meas_layout.addLayout(additionalLayout)
        self.add_distance_btn.clicked.connect(self.handle_add_distance)
        self.add_count_btn.clicked.connect(self.handle_add_count)

        # --- Progress indicator ---
        progressLayout = QHBoxLayout()
        self.progressBar = QProgressBar()
        self.progressBar.setMinimum(0)
        self.progressBar.setMaximum(100)
        self.timeRemainingLabel = QLabel("Estimated time: N/A")
        progressLayout.addWidget(self.progressBar)
        progressLayout.addWidget(self.timeRemainingLabel)
        meas_layout.addLayout(progressLayout)

        # --- Measurement log (optional visibility) ---
        logLayout = QVBoxLayout()
        self.logCheckBox = QCheckBox("Show log")
        self.logCheckBox.setChecked(True)
        logLayout.addWidget(self.logCheckBox)
        self.measurementLog = QPlainTextEdit()
        self.measurementLog.setReadOnly(True)
        self.measurementLog.setMaximumBlockCount(2000)  # prevent memory bloat
        logLayout.addWidget(self.measurementLog)
        self.logCheckBox.toggled.connect(self.measurementLog.setVisible)
        self.measurementLog.setVisible(self.logCheckBox.isChecked())
        meas_layout.addLayout(logLayout)

        # Timer for stage XY updates
        from PyQt5.QtCore import QTimer

        self.xyTimer = QTimer(self)
        self.xyTimer.timeout.connect(self.update_xy_pos)
        self.xyTimer.start(10000)
