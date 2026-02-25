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
        
        # Set smaller font for all controls to fit smaller screens
        try:
            from PyQt5.QtGui import QFont
            control_font = QFont()
            control_font.setPointSize(9)  # Smaller font for controls (menu-size)
            meas_tab.setFont(control_font)
        except Exception:
            pass
        
        meas_layout = QVBoxLayout(meas_tab)
        # Reduce vertical spacing for compact layout
        meas_layout.setContentsMargins(4, 2, 4, 2)  # Tight margins
        meas_layout.setSpacing(2)  # Minimal spacing between rows
        self.tabs.addTab(meas_tab, "Measurements")

        # --- Measurement controls ---
        buttonLayout = QHBoxLayout()
        buttonLayout.setSpacing(4)  # Compact spacing
        self.initializeBtn = QPushButton("Initialize Hardware")
        self.initializeBtn.clicked.connect(self.toggle_hardware)
        try:
            self.initializeBtn.setMaximumHeight(28)  # Compact button height
        except Exception:
            pass
        self.start_btn = QPushButton("Start measurement")
        self.start_btn.clicked.connect(self.start_measurements)
        try:
            self.start_btn.setMaximumHeight(28)
        except Exception:
            pass
        self.pause_btn = QPushButton("Pause")
        self.pause_btn.clicked.connect(self.pause_measurements)
        try:
            self.pause_btn.setMaximumHeight(28)
        except Exception:
            pass
        self.skip_btn = QPushButton("Skip")
        self.skip_btn.clicked.connect(self.skip_current_point)
        try:
            self.skip_btn.setMaximumHeight(28)
        except Exception:
            pass
        self.stop_btn = QPushButton("Stop")
        self.stop_btn.clicked.connect(self.stop_measurements)
        try:
            self.stop_btn.setMaximumHeight(28)
        except Exception:
            pass
        self.start_btn.setEnabled(False)
        self.pause_btn.setEnabled(False)
        self.skip_btn.setEnabled(False)
        self.stop_btn.setEnabled(False)
        self._sidecar_locked = False
        self._sidecar_alive = False
        self._sidecar_lock_reason = ""
        buttonLayout.addWidget(self.initializeBtn)
        buttonLayout.addWidget(self.start_btn)
        buttonLayout.addWidget(self.pause_btn)
        buttonLayout.addWidget(self.skip_btn)
        buttonLayout.addWidget(self.stop_btn)
        meas_layout.addLayout(buttonLayout)

        # --- Hardware status indicators ---
        statusLayout = QHBoxLayout()
        statusLayout.setSpacing(4)  # Compact spacing
        xyLabel = QLabel("XY Stage:")
        self.xyStageIndicator = QLabel()
        self.xyStageIndicator.setFixedSize(16, 16)  # Smaller indicators
        self.xyStageIndicator.setStyleSheet(
            "background-color: gray; border-radius: 8px;"
        )
        statusLayout.addWidget(xyLabel)
        statusLayout.addWidget(self.xyStageIndicator)
        cameraLabel = QLabel("Camera:")
        self.cameraIndicator = QLabel()
        self.cameraIndicator.setFixedSize(16, 16)  # Smaller indicators
        self.cameraIndicator.setStyleSheet(
            "background-color: gray; border-radius: 8px;"
        )
        statusLayout.addWidget(cameraLabel)
        statusLayout.addWidget(self.cameraIndicator)
        sidecarLabel = QLabel("A2K Sidecar:")
        self.sidecarIndicator = QLabel()
        self.sidecarIndicator.setFixedSize(16, 16)
        self.sidecarIndicator.setStyleSheet(
            "background-color: gray; border-radius: 8px;"
        )
        self.sidecarStatusLabel = QLabel("N/A")
        try:
            self.sidecarStatusLabel.setStyleSheet("color: #666; font-size: 10px;")
        except Exception:
            pass
        statusLayout.addWidget(sidecarLabel)
        statusLayout.addWidget(self.sidecarIndicator)
        statusLayout.addWidget(self.sidecarStatusLabel)
        statusLayout.addStretch()
        self.homeBtn = QPushButton("Home")
        self.homeBtn.clicked.connect(self.home_stage_button_clicked)
        try:
            self.homeBtn.setMaximumHeight(28)
        except Exception:
            pass
        statusLayout.addWidget(self.homeBtn)
        self.loadPosBtn = QPushButton("Load Position")
        self.loadPosBtn.clicked.connect(self.load_position_button_clicked)
        try:
            self.loadPosBtn.setMaximumHeight(28)
        except Exception:
            pass
        statusLayout.addWidget(self.loadPosBtn)
        meas_layout.addLayout(statusLayout)

        # --- Stage position controls ---
        posLayout = QHBoxLayout()
        posLayout.setSpacing(4)  # Compact spacing
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

        # --- Current position display ---
        currentPosLayout = QHBoxLayout()
        self.currentPositionLabel = QLabel("Current XY: (Not initialized)")
        self.currentPositionLabel.setStyleSheet(
            "color: #666; font-size: 10px; margin: 2px;"
        )
        currentPosLayout.addWidget(self.currentPositionLabel)
        currentPosLayout.addStretch()  # Push to the left
        meas_layout.addLayout(currentPosLayout)

        # --- Integration + Attenuation ---
        integrationLayout = QHBoxLayout()
        integrationLayout.setSpacing(4)  # Compact spacing
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

        # Integration settings (no SAXS/WAXS quick distance buttons)
        meas_layout.addLayout(integrationLayout)

        # --- Folder selection ---
        folderLayout = QHBoxLayout()
        folderLayout.setSpacing(4)  # Compact spacing
        folderLabel = QLabel("Save Folder:")
        self.folderLineEdit = QLineEdit()
        # Get measurements folder from config (same as session manager)
        default_folder = ""
        if hasattr(self, "config") and self.config:
            default_folder = self.config.get("measurements_folder", 
                            self.config.get("default_folder", ""))
        self.folderLineEdit.setText(default_folder)
        # Connect to update when changed
        self.folderLineEdit.editingFinished.connect(self._on_folder_changed)
        self.browseBtn = QPushButton("Browse...")
        self.browseBtn.clicked.connect(self.browse_folder)
        folderLayout.addWidget(folderLabel)
        folderLayout.addWidget(self.folderLineEdit)
        folderLayout.addWidget(self.browseBtn)
        meas_layout.addLayout(folderLayout)

        # --- Sample ID ---
        fileNameLayout = QHBoxLayout()
        fileNameLayout.setSpacing(4)  # Compact spacing
        fileNameLabel = QLabel("Sample ID:")
        self.fileNameLineEdit = QLineEdit()
        # Connect to update session when changed
        self.fileNameLineEdit.editingFinished.connect(self._on_sample_id_changed)
        fileNameLayout.addWidget(fileNameLabel)
        fileNameLayout.addWidget(self.fileNameLineEdit)
        
        # Add lock indicator
        self.sampleIdLockLabel = QLabel("")
        self.sampleIdLockLabel.setStyleSheet("color: #888; font-size: 9px;")
        fileNameLayout.addWidget(self.sampleIdLockLabel)
        
        meas_layout.addLayout(fileNameLayout)

        # --- Additional controls for count and distance ---
        additionalLayout = QHBoxLayout()
        additionalLayout.setSpacing(4)  # Compact spacing

        # Add count controls
        self.add_count_btn = QPushButton("Add count")
        self.addCountSpinBox = QSpinBox()
        self.addCountSpinBox.setMinimum(1)
        self.addCountSpinBox.setMaximum(10000)
        self.addCountSpinBox.setValue(60)
        additionalLayout.addWidget(self.add_count_btn)
        additionalLayout.addWidget(self.addCountSpinBox)
        self.add_count_btn.clicked.connect(self.handle_add_count)

        # Add configurable distance buttons
        self._create_distance_buttons(additionalLayout)

        meas_layout.addLayout(additionalLayout)

        # --- Progress indicator ---
        progressLayout = QHBoxLayout()
        progressLayout.setSpacing(4)  # Compact spacing
        self.progressBar = QProgressBar()
        self.progressBar.setMinimum(0)
        self.progressBar.setMaximum(100)
        self.timeRemainingLabel = QLabel("Estimated time: N/A")
        progressLayout.addWidget(self.progressBar)
        progressLayout.addWidget(self.timeRemainingLabel)
        meas_layout.addLayout(progressLayout)

        # --- Measurement log (optional visibility) ---
        logLayout = QVBoxLayout()
        logLayout.setSpacing(2)  # Minimal spacing
        self.logCheckBox = QCheckBox("Show log")
        self.logCheckBox.setChecked(True)
        logLayout.addWidget(self.logCheckBox)
        self.measurementLog = QPlainTextEdit()
        self.measurementLog.setReadOnly(True)
        self.measurementLog.setMaximumBlockCount(2000)  # prevent memory bloat
        
        # Set compact maximum height for log - user can expand if needed
        try:
            self.measurementLog.setMaximumHeight(80)  # Show ~4-5 lines by default
        except Exception:
            pass
        
        logLayout.addWidget(self.measurementLog)
        self.logCheckBox.toggled.connect(self.measurementLog.setVisible)
        self.measurementLog.setVisible(self.logCheckBox.isChecked())
        meas_layout.addLayout(logLayout)

        # Timer for stage XY updates
        from PyQt5.QtCore import QTimer

        self.xyTimer = QTimer(self)
        self.xyTimer.timeout.connect(self.update_xy_pos)
        self.xyTimer.start(500)
        self.sidecarHeartbeatTimer = QTimer(self)
        self.sidecarHeartbeatTimer.timeout.connect(self.refresh_sidecar_status)
        self.sidecarHeartbeatTimer.start(1000)
        self.update_xy_pos()
        self.refresh_sidecar_status()
        QTimer.singleShot(300, self.sync_hardware_state_from_backend)

    def _create_distance_buttons(self, layout):
        """
        Create distance buttons based on configuration.
        Each button appends its configured text to the filename when clicked.
        """
        self.distance_buttons = []

        # Get distance buttons config from global config
        distance_buttons_config = []
        if hasattr(self, "config"):
            distance_buttons_config = self.config.get("distance_buttons", [])

        # Default fallback if no config
        if not distance_buttons_config:
            distance_buttons_config = [
                {"text": "+2cm", "append_text": "_2cm"},
                {"text": "+17cm", "append_text": "_17cm"},
            ]

        # Create buttons dynamically
        for button_config in distance_buttons_config:
            button_text = button_config.get("text", "Distance")
            append_text = button_config.get("append_text", "_dist")

            btn = QPushButton(button_text)
            btn.clicked.connect(
                lambda checked, text=append_text: self._handle_distance_button_click(
                    text
                )
            )
            layout.addWidget(btn)
            self.distance_buttons.append(btn)

    def _on_sample_id_changed(self):
        """Handle Sample ID field change - update session container if active and unlocked."""
        try:
            if not hasattr(self, 'session_manager') or not self.session_manager.is_session_active():
                return
            
            new_sample_id = self.fileNameLineEdit.text().strip()
            if not new_sample_id:
                return
            
            # Check if locked
            if self.session_manager.is_locked():
                self.sampleIdLockLabel.setText("🔒 Locked")
                self.sampleIdLockLabel.setStyleSheet("color: #d00; font-size: 9px;")
                return
            
            # Update in container
            if self.session_manager.update_sample_id(new_sample_id):
                self.sampleIdLockLabel.setText("✓ Updated")
                self.sampleIdLockLabel.setStyleSheet("color: #0a0; font-size: 9px;")
                
                # Update window title if method exists
                if hasattr(self, 'update_session_status'):
                    self.update_session_status()
            else:
                self.sampleIdLockLabel.setText("✗ Failed")
                self.sampleIdLockLabel.setStyleSheet("color: #d00; font-size: 9px;")
                
        except Exception as e:
            import logging
            logger = logging.getLogger(__name__)
            logger.warning(f"Failed to update sample ID: {e}")
    
    def _on_folder_changed(self):
        """Handle folder change - just log for now, actual path used at measurement time."""
        try:
            new_folder = self.folderLineEdit.text().strip()
            if new_folder:
                import logging
                from pathlib import Path
                logger = logging.getLogger(__name__)
                
                # Create folder if it doesn't exist
                folder_path = Path(new_folder)
                folder_path.mkdir(parents=True, exist_ok=True)
                
                logger.info(f"Measurement folder updated: {new_folder}")
        except Exception as e:
            import logging
            logger = logging.getLogger(__name__)
            logger.warning(f"Failed to update folder: {e}")
    
    def _handle_distance_button_click(self, append_text):
        """
        Handle distance button click by appending text to sample ID.
        """
        current_filename = self.fileNameLineEdit.text()
        self.fileNameLineEdit.setText(current_filename + append_text)
