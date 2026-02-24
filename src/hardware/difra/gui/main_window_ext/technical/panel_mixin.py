import logging

from hardware.difra.gui.main_window_ext.technical.helpers import _get_default_folder

logger = logging.getLogger(__name__)


def _tm():
    from hardware.difra.gui.main_window_ext import technical_measurements as tm

    return tm


class TechnicalPanelMixin:
    def create_technical_panel(self):
        tm = _tm()
        self.aux_counter = 0
        super().create_zone_measurements()

        self.continuous_movement_controller = None
        self._initialize_continuous_movement_controller()

        title = "Technical Measurements"
        self.measDock = tm.QDockWidget(title, self)
        self.measDock.setObjectName("TechnicalMeasurementsDock")
        self.measDock.setAllowedAreas(tm.Qt.LeftDockWidgetArea | tm.Qt.RightDockWidgetArea)

        container = tm.QWidget()
        outer = tm.QVBoxLayout(container)
        outer.setContentsMargins(6, 4, 6, 4)
        outer.setSpacing(6)

        try:
            from PyQt5.QtGui import QFont

            control_font = QFont()
            control_font.setPointSize(9)
            container.setFont(control_font)
        except Exception:
            pass

        it_layout = tm.QHBoxLayout()
        it_layout.addWidget(tm.QLabel("Integration Time (s):"))
        self.integrationTimeSpin = tm.QDoubleSpinBox()
        self.integrationTimeSpin.setDecimals(6)
        self.integrationTimeSpin.setRange(1e-6, 1e4)
        self.integrationTimeSpin.setSingleStep(1e-6)
        self.integrationTimeSpin.setValue(1.0)
        it_layout.addWidget(self.integrationTimeSpin)

        it_layout.addWidget(tm.QLabel("Frames:"))
        self.captureFramesSpin = tm.QSpinBox()
        self.captureFramesSpin.setRange(1, 1_000_000)
        self.captureFramesSpin.setValue(1)
        self.captureFramesSpin.setToolTip(
            "Capture N frames at the given integration time; frames will be averaged into a single final image"
        )
        it_layout.addWidget(self.captureFramesSpin)

        outer.addLayout(it_layout)

        cm_layout = tm.QHBoxLayout()
        self.moveContinuousCheck = tm.QCheckBox("Move Continuous (AgBH)")
        self.moveContinuousCheck.setToolTip(
            "Enable continuous circular movement during AgBH measurements to smooth out sample inconsistencies"
        )
        cm_layout.addWidget(self.moveContinuousCheck)

        cm_layout.addWidget(tm.QLabel("Radius (mm):"))
        self.movementRadiusSpin = tm.QDoubleSpinBox()
        self.movementRadiusSpin.setRange(0.1, 10.0)
        self.movementRadiusSpin.setSingleStep(0.1)
        self.movementRadiusSpin.setValue(2.0)
        self.movementRadiusSpin.setDecimals(1)
        self.movementRadiusSpin.setToolTip(
            "Maximum radius for continuous movement pattern (decreases during measurement)"
        )
        cm_layout.addWidget(self.movementRadiusSpin)

        outer.addLayout(cm_layout)

        fld = tm.QHBoxLayout()
        fld.addWidget(tm.QLabel("Save Folder:"))
        self.folderLE = tm.QLineEdit()
        default_folder = _get_default_folder(self.config if hasattr(self, "config") else None)
        self.folderLE.setText(default_folder)

        fld.addWidget(self.folderLE, 1)
        b = tm.QPushButton("Browse…")
        b.clicked.connect(self._browse_folder)
        fld.addWidget(b)
        outer.addLayout(fld)

        row = tm.QHBoxLayout()
        self.auxBtn = tm.QPushButton("Measure")
        self.auxBtn.clicked.connect(self.measure_aux)
        row.addWidget(self.auxBtn)

        self._aux_status = tm.QLabel("")
        row.addWidget(self._aux_status)
        self._aux_timer = tm.QTimer(self)
        self._aux_timer.setInterval(200)
        self._aux_timer.timeout.connect(self._update_aux_status)

        self.auxNameLE = tm.QLineEdit()
        self.auxNameLE.setPlaceholderText("Measurement name (for metadata generation)")
        self.auxNameLE.setToolTip(
            "Enter name for auxiliary measurement - used for metadata file generation"
        )
        row.addWidget(self.auxNameLE, 1)
        outer.addLayout(row)

        self.auxTable = tm.QTableWidget()
        self.auxTable.setColumnCount(4)
        self.auxTable.installEventFilter(self)
        self.auxTable.setHorizontalHeaderLabels(["Primary", "File", "Type", "Alias"])

        self.auxTable.verticalHeader().setVisible(False)
        self.auxTable.setAlternatingRowColors(True)
        try:
            from PyQt5.QtGui import QFont
            from PyQt5.QtWidgets import QHeaderView

            header = self.auxTable.horizontalHeader()
            header.setSectionResizeMode(0, QHeaderView.Fixed)
            header.setSectionResizeMode(1, QHeaderView.Stretch)
            header.setSectionResizeMode(2, QHeaderView.Fixed)
            header.setSectionResizeMode(3, QHeaderView.Fixed)

            self.auxTable.setColumnWidth(0, 60)
            self.auxTable.setColumnWidth(2, 60)
            self.auxTable.setColumnWidth(3, 60)

            font = QFont()
            font.setPointSize(9)
            self.auxTable.setFont(font)

            self.auxTable.verticalHeader().setDefaultSectionSize(24)
        except Exception:
            pass
        self.auxTable.setSelectionBehavior(self.auxTable.SelectRows)
        self.auxTable.setSelectionMode(self.auxTable.ExtendedSelection)
        self.auxTable.cellDoubleClicked.connect(self._open_measurement_from_table)
        outer.addWidget(self.auxTable)

        actions_layout = tm.QHBoxLayout()
        actions_layout.setContentsMargins(0, 0, 0, 0)
        actions_layout.setSpacing(4)
        actions_layout.addWidget(tm.QLabel("Actions:"))

        self.load_h5_btn = tm.QPushButton("Load Container…")
        self.load_h5_btn.setToolTip("Load an existing technical HDF5 container")
        self.load_h5_btn.clicked.connect(self.load_technical_h5)
        actions_layout.addWidget(self.load_h5_btn)

        self.dist_btn = tm.QPushButton("Distances...")
        self.dist_btn.setToolTip("Configure detector distances for technical measurements")
        self.dist_btn.clicked.connect(self.configure_detector_distances)
        actions_layout.addWidget(self.dist_btn)

        self.lock_h5_btn = tm.QPushButton("Lock Container")
        self.lock_h5_btn.setToolTip("Lock active technical container for production use")
        self.lock_h5_btn.clicked.connect(self.lock_active_technical_container)
        actions_layout.addWidget(self.lock_h5_btn)

        self.pyfai_btn = tm.QPushButton("PyFAI")
        self.pyfai_btn.setToolTip("Run pyfai-calib2 in this folder")
        self.pyfai_btn.clicked.connect(self.run_pyfai)
        actions_layout.addWidget(self.pyfai_btn)

        outer.addLayout(actions_layout)

        rt_layout = tm.QHBoxLayout()
        rt_layout.addWidget(tm.QLabel("Frames/⟳:"))
        self.framesSpin = tm.QSpinBox()
        self.framesSpin.setRange(1, 1_000_000)
        self.framesSpin.setValue(1)
        rt_layout.addWidget(self.framesSpin)

        self.rtBtn = tm.QPushButton("Real-time")
        self.rtBtn.setCheckable(True)
        self.rtBtn.clicked.connect(self._toggle_realtime)
        rt_layout.addWidget(self.rtBtn)

        outer.addLayout(rt_layout)

        scroll = tm.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(container)
        self.measDock.setWidget(scroll)

        try:
            self.measDock.setMinimumWidth(300)
        except Exception:
            pass

        self.addDockWidget(tm.Qt.LeftDockWidgetArea, self.measDock)

        self.enable_measurement_controls(False)
        self.hardware_state_changed.connect(self.enable_measurement_controls)
        self.hardware_state_changed.connect(lambda _: self.refresh_aux_table_alias_models())
        self.hardware_state_changed.connect(lambda _: self._initialize_continuous_movement_controller())

    def enable_measurement_controls(self, enable: bool):
        status = "enabled" if enable else "disabled"
        self._log_technical_event(f"Technical measurement controls {status}")
        widgets = [
            self.integrationTimeSpin,
            self.captureFramesSpin,
            self.moveContinuousCheck,
            self.movementRadiusSpin,
            self.folderLE,
            self.auxNameLE,
            self.auxTable,
            self.framesSpin,
            self.rtBtn,
        ]
        if hasattr(self, "load_h5_btn"):
            self.load_h5_btn.setEnabled(True)
        if hasattr(self, "lock_h5_btn"):
            self.lock_h5_btn.setEnabled(True)

        for w in widgets:
            w.setEnabled(enable)

        self._update_distance_dependent_controls()

    def _update_distance_dependent_controls(self):
        has_distances = hasattr(self, "_detector_distances") and bool(self._detector_distances)

        if hasattr(self, "auxBtn"):
            self.auxBtn.setEnabled(has_distances)
        if hasattr(self, "pyfai_btn"):
            self.pyfai_btn.setEnabled(has_distances)
        if hasattr(self, "lock_h5_btn"):
            self.lock_h5_btn.setEnabled(has_distances)

        if not has_distances and hasattr(self, "auxBtn"):
            self.auxBtn.setStyleSheet("color: gray;")
        elif hasattr(self, "auxBtn"):
            self.auxBtn.setStyleSheet("")

    def _update_window_title_with_distances(self):
        if not hasattr(self, "_detector_distances") or not self._detector_distances:
            return

        detector_configs = self.config.get("detectors", [])
        distance_parts = []
        for detector_id, distance_cm in sorted(self._detector_distances.items()):
            detector_config = next((d for d in detector_configs if d.get("id") == detector_id), None)
            alias = detector_config.get("alias", detector_id) if detector_config else detector_id
            distance_parts.append(f"{alias}: {distance_cm} cm")

        distance_str = ", ".join(distance_parts)
        base_title = "EosDX Scanning Software"
        is_dev = self.config.get("DEV", False)
        dev_suffix = " [DEMO]" if is_dev else ""

        new_title = f"{base_title} ({distance_str}){dev_suffix}"
        self.setWindowTitle(new_title)
        self._log_technical_event(f"Window title updated with distances: {distance_str}")

    def configure_detector_distances(self):
        tm = _tm()
        from hardware.difra.gui.detector_distance_config_dialog import DetectorDistanceConfigDialog

        detector_configs = self.config.get("detectors", [])
        if not detector_configs:
            tm.QMessageBox.warning(
                self,
                "No Detectors",
                "No detector configuration found.",
            )
            return

        active_detector_ids = self._get_active_detector_ids()
        active_detector_configs = [d for d in detector_configs if d.get("id") in active_detector_ids]
        if not active_detector_configs:
            tm.QMessageBox.warning(
                self,
                "No Active Detectors",
                "No active detectors configured.",
            )
            return

        current_distances = getattr(self, "_detector_distances", {})
        dialog = DetectorDistanceConfigDialog(
            detector_configs=active_detector_configs,
            current_distances=current_distances,
            parent=self,
        )

        if dialog.exec_() == tm.QDialog.Accepted:
            distances = dialog.get_distances()
            self._detector_distances = distances

            dist_str = ", ".join(
                f"{d.get('alias', d['id'])}: {distances.get(d['id'], 'N/A')} cm"
                for d in active_detector_configs
            )
            self._log_technical_event(f"Configured distances: {dist_str}")
            self._update_window_title_with_distances()
            self._update_distance_dependent_controls()
            if hasattr(self, "_on_detector_distances_updated"):
                try:
                    self._on_detector_distances_updated()
                except Exception as exc:
                    logger.warning("Distance update hook failed: %s", exc, exc_info=True)

            tm.QMessageBox.information(
                self,
                "Distances Configured",
                f"Detector distances set:\n\n{dist_str}",
            )

    def _initialize_continuous_movement_controller(self):
        try:
            from hardware.difra.gui.technical.continuous_movement import ContinuousMovementController

            stage_controller = None
            if hasattr(self, "hardware_controller") and self.hardware_controller:
                stage_controller = self.hardware_controller.stage_controller
            elif hasattr(self, "stage_controller"):
                stage_controller = self.stage_controller
            elif hasattr(self, "hardware_client") and self.hardware_client:
                stage_controller = self.hardware_client.stage_controller

            if stage_controller:
                self.continuous_movement_controller = ContinuousMovementController(
                    stage_controller=stage_controller, parent=self
                )
                self.continuous_movement_controller.movement_error.connect(
                    lambda msg: self._log_technical_event(f"Movement error: {msg}")
                )
                self._log_technical_event("Continuous movement controller initialized")
                logger.info("Continuous movement controller initialized")
            else:
                self._log_technical_event("No stage controller available for continuous movement")
                logger.debug("No stage controller available for continuous movement")
        except ImportError as e:
            logger.warning(f"Failed to import continuous movement controller: {e}")
        except Exception as e:
            logger.error(f"Error initializing continuous movement controller: {e}", exc_info=True)

    def _browse_folder(self):
        tm = _tm()
        f = tm.QFileDialog.getExistingDirectory(self, "Select Folder")
        if f:
            self.folderLE.setText(f)

    def initialize_hardware(self):
        pass
