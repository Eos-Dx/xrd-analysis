import json
import logging
import os
import queue
import re
import subprocess
import sys
import time
import traceback
import uuid

import matplotlib.pyplot as plt
import numpy as np

# Module logger
logger = logging.getLogger(__name__)

# Robust Qt imports to allow tests to run without a full PyQt5 installation
try:
    from PyQt5.QtCore import QEvent, Qt, QThread, QTimer
    from PyQt5.QtWidgets import (
        QCheckBox,
        QComboBox,
        QDialog,
        QDockWidget,
        QDoubleSpinBox,
        QFileDialog,
        QFormLayout,
        QGroupBox,
        QHBoxLayout,
        QInputDialog,
        QLabel,
        QLineEdit,
        QMessageBox,
        QPushButton,
        QScrollArea,
        QSpinBox,
        QTableWidget,
        QTableWidgetItem,
        QVBoxLayout,
        QWidget,
    )
except Exception:  # pragma: no cover - test stubs
    import types

    class _Stub:
        def __init__(self, *a, **k):
            pass

    # Minimal QtCore stubs
    QEvent = object

    class _Qt:
        LeftDockWidgetArea = 1
        RightDockWidgetArea = 2
        Horizontal = 0
        Key_Delete = 16777223
        UserRole = 32

    Qt = _Qt()

    class _QThread:
        def __init__(self, *a, **k):
            pass

        def start(self):
            pass

        def quit(self):
            pass

        def deleteLater(self):
            pass

    QThread = _QThread

    class _QTimer:
        def __init__(self, *a, **k):
            pass

        def setInterval(self, *a, **k):
            pass

        def timeout(self, *a, **k):
            return types.SimpleNamespace(connect=lambda *a, **k: None)

        def start(self, *a, **k):
            pass

    QTimer = _QTimer

    # Minimal QtWidgets stubs used by this module
    QCheckBox = _Stub
    QComboBox = _Stub

    class QDialog(_Stub):
        Accepted = 1
        Rejected = 0

    QDockWidget = _Stub
    QDoubleSpinBox = _Stub
    QFileDialog = _Stub
    QFormLayout = _Stub
    QGroupBox = _Stub
    QHBoxLayout = _Stub
    QInputDialog = _Stub
    QLabel = _Stub
    QLineEdit = _Stub
    QPushButton = _Stub
    QScrollArea = _Stub
    QSpinBox = _Stub
    QTableWidget = _Stub
    QTableWidgetItem = _Stub
    QVBoxLayout = _Stub
    QWidget = _Stub

    class QMessageBox:
        Yes, No = 1, 0

        @staticmethod
        def warning(*args, **kwargs):
            return None

        @staticmethod
        def question(*args, **kwargs):
            return QMessageBox.Yes

        @staticmethod
        def critical(*args, **kwargs):
            return None

        @staticmethod
        def information(*args, **kwargs):
            return None


# Avoid importing heavy zone_measurements dependencies during tests
try:
    from hardware.difra.gui.main_window_ext.zone_measurements import (
        ZoneMeasurementsMixin as _ZoneMeasurementsMixin,
    )
except Exception:  # pragma: no cover - test stubs

    class _ZoneMeasurementsMixin(object):
        pass


# Defer all technical imports to avoid pyFAI crashes on startup
# These will be imported only when actually needed
_TECHNICAL_IMPORTS_AVAILABLE = None  # None = not yet tested
_technical_modules = {}

def _get_technical_imports():
    """Lazy import of technical modules to avoid startup crashes."""
    global _TECHNICAL_IMPORTS_AVAILABLE, _technical_modules
    
    if _TECHNICAL_IMPORTS_AVAILABLE is not None:
        return _TECHNICAL_IMPORTS_AVAILABLE
    
    try:
        from hardware.difra.gui.technical.capture import (
            CaptureWorker,
            show_measurement_window,
            validate_folder,
        )
        from hardware.difra.gui.technical.measurement_worker import MeasurementWorker
        
        _technical_modules.update({
            'CaptureWorker': CaptureWorker,
            'show_measurement_window': show_measurement_window,
            'validate_folder': validate_folder,
            'MeasurementWorker': MeasurementWorker,
        })
        _TECHNICAL_IMPORTS_AVAILABLE = True
        logger.info("Technical measurement imports successful")
        return True
    except Exception as e:
        # Get detailed traceback for debugging
        tb_str = ''.join(traceback.format_exception(type(e), e, e.__traceback__))
        logger.error(
            f"Technical measurement imports failed: {type(e).__name__}: {e}\n{tb_str}",
            exc_info=True
        )
        _TECHNICAL_IMPORTS_AVAILABLE = False
        return False

def _get_technical_temp_folder(config=None):
    """Get the technical temp folder path with platform-specific defaults.
    
    Args:
        config: Optional config dict from global.json
    
    Returns:
        Path to technical temp folder (created if it doesn't exist)
    """
    from pathlib import Path
    import platform
    import tempfile
    
    # Check config first
    if config and config.get("technical_temp_folder"):
        temp_path = Path(config["technical_temp_folder"])
    else:
        # Platform-specific defaults
        system = platform.system()
        if system == "Darwin":  # macOS
            temp_path = Path.home() / "dev" / "Data" / "tech_temp"
        elif system == "Windows":
            temp_path = Path("C:/dev/Data/tech_temp")
        else:  # Linux or other
            temp_path = Path.home() / "dev" / "Data" / "tech_temp"
    
    # Create directory if it doesn't exist
    try:
        temp_path.mkdir(parents=True, exist_ok=True)
        logger.debug(f"Technical temp folder: {temp_path}")
        return str(temp_path)
    except Exception as e:
        # Fallback to system temp if creation fails
        logger.warning(f"Failed to create technical temp folder {temp_path}: {e}")
        fallback = Path(tempfile.gettempdir()) / "difra_technical"
        fallback.mkdir(parents=True, exist_ok=True)
        logger.info(f"Using fallback temp folder: {fallback}")
        return str(fallback)


def _get_archive_base_folder(config=None):
    """Get the base archive folder path with platform-specific defaults.
    
    Args:
        config: Optional config dict from global.json
    
    Returns:
        Path to archive base folder (created if it doesn't exist)
    """
    from pathlib import Path
    import platform
    
    # Check config first
    if config and config.get("archive_base_folder"):
        archive_path = Path(config["archive_base_folder"])
    else:
        # Platform-specific defaults: ~/dev/Data/archive
        system = platform.system()
        if system == "Darwin":  # macOS
            archive_path = Path.home() / "dev" / "Data" / "archive"
        elif system == "Windows":
            archive_path = Path("C:/dev/Data/archive")
        else:  # Linux or other
            archive_path = Path.home() / "dev" / "Data" / "archive"
    
    # Create directory if it doesn't exist
    try:
        archive_path.mkdir(parents=True, exist_ok=True)
        logger.debug(f"Archive base folder: {archive_path}")
        return str(archive_path)
    except Exception as e:
        logger.warning(f"Failed to create archive base folder {archive_path}: {e}")
        # Fallback to home directory
        return str(Path.home())


def _get_technical_storage_folder(config=None):
    """Get the technical storage folder path (archive/technical).
    
    Args:
        config: Optional config dict from global.json
    
    Returns:
        Path to technical storage folder (created if it doesn't exist)
    """
    from pathlib import Path
    
    # Get base archive folder and append 'technical' subfolder
    archive_base = _get_archive_base_folder(config)
    storage_path = Path(archive_base) / "technical"
    
    # Create directory if it doesn't exist
    try:
        storage_path.mkdir(parents=True, exist_ok=True)
        logger.debug(f"Technical storage folder: {storage_path}")
        return str(storage_path)
    except Exception as e:
        logger.warning(f"Failed to create technical storage folder {storage_path}: {e}")
        # Fallback to temp folder
        return _get_technical_temp_folder(config)


def _get_measurements_archive_folder(config=None):
    """Get the measurements archive folder path (archive/measurements).
    
    Args:
        config: Optional config dict from global.json
    
    Returns:
        Path to measurements archive folder (created if it doesn't exist)
    """
    from pathlib import Path
    
    # Get base archive folder and append 'measurements' subfolder
    archive_base = _get_archive_base_folder(config)
    archive_path = Path(archive_base) / "measurements"
    
    # Create directory if it doesn't exist
    try:
        archive_path.mkdir(parents=True, exist_ok=True)
        logger.debug(f"Measurements archive folder: {archive_path}")
        return str(archive_path)
    except Exception as e:
        logger.warning(f"Failed to create measurements archive folder {archive_path}: {e}")
        # Fallback to home directory
        return str(Path.home())


def _get_measurement_default_folder(config=None):
    """Get the measurement default folder path with platform-specific defaults.
    
    Args:
        config: Optional config dict from global.json
    
    Returns:
        Path to measurement default folder (created if it doesn't exist)
    """
    from pathlib import Path
    import platform
    
    # Check config first
    if config and config.get("measurement_default_folder"):
        meas_path = Path(config["measurement_default_folder"])
    else:
        # Platform-specific defaults
        system = platform.system()
        if system == "Darwin":  # macOS
            meas_path = Path.home() / "dev" / "Data" / "measurements"
        elif system == "Windows":
            meas_path = Path("C:/dev/Data/measurements")
        else:  # Linux or other
            meas_path = Path.home() / "dev" / "Data" / "measurements"
    
    # Create directory if it doesn't exist
    try:
        meas_path.mkdir(parents=True, exist_ok=True)
        logger.debug(f"Measurement default folder: {meas_path}")
        return str(meas_path)
    except Exception as e:
        logger.warning(f"Failed to create measurement default folder {meas_path}: {e}")
        # Fallback to home directory
        return str(Path.home())


def _get_default_folder(config=None):
    """Get the default folder for technical measurements UI.
    
    Returns the technical storage folder as the default.
    
    Args:
        config: Optional config dict from global.json
    
    Returns:
        Path to default folder
    """
    # Check explicit default_folder in config
    if config and config.get("default_folder"):
        return config["default_folder"]
    
    # Otherwise use technical storage folder
    return _get_technical_storage_folder(config)


def _get_technical_module(name):
    """Get a technical module by name, with fallback stubs."""
    if _get_technical_imports():
        return _technical_modules.get(name)
    else:
        # Return stub implementations
        stubs = {
            'CaptureWorker': type('CaptureWorker', (), {
                '__init__': lambda self, *args, **kwargs: None,
                'moveToThread': lambda self, thread: None,
                'finished': type('Signal', (), {'connect': lambda self, f: None})()
            }),
            'show_measurement_window': lambda *args, **kwargs: print("Technical measurement window not available - imports failed"),
            'validate_folder': lambda path: str(path) if path else "",
            'MeasurementWorker': type('MeasurementWorker', (), {
                '__init__': lambda self, *args, **kwargs: None,
                'run': lambda self: None,
                'add_aux_item': type('Signal', (), {'connect': lambda self, f: None})()
            })
        }
        return stubs.get(name)


class PoniFileSelectionDialog(QDialog):
    """Dialog for selecting PONI files for each detector alias."""

    def __init__(self, aliases, current_poni_files=None, parent=None):
        super().__init__(parent)
        self.aliases = aliases
        self.poni_files = {}
        self.line_edits = {}

        # Pre-populate with current PONI files if available
        if current_poni_files:
            for alias in aliases:
                if alias in current_poni_files:
                    poni_info = current_poni_files[alias]
                    if isinstance(poni_info, dict) and "path" in poni_info:
                        self.poni_files[alias] = poni_info["path"]
                    elif (
                        hasattr(self.parent(), "poni_files")
                        and alias in self.parent().poni_files
                    ):
                        # Fallback to parent's poni_files if available
                        parent_poni = self.parent().poni_files[alias]
                        if isinstance(parent_poni, dict) and "path" in parent_poni:
                            self.poni_files[alias] = parent_poni["path"]

        self.setup_ui()

    def setup_ui(self):
        self.setWindowTitle("Select PONI Files for Technical Meta")
        self.setModal(True)
        self.resize(600, 400)

        layout = QVBoxLayout(self)

        # Header
        header = QLabel("Select PONI calibration files for each detector alias:")
        header.setStyleSheet("font-weight: bold; margin-bottom: 10px;")
        layout.addWidget(header)

        # Form layout for PONI file selection
        form_layout = QFormLayout()

        for alias in self.aliases:
            # Create horizontal layout for each alias
            h_layout = QHBoxLayout()

            # Line edit for file path
            line_edit = QLineEdit()
            line_edit.setPlaceholderText(f"Select PONI file for {alias}")
            if alias in self.poni_files:
                line_edit.setText(self.poni_files[alias])
            self.line_edits[alias] = line_edit
            h_layout.addWidget(line_edit)

            # Browse button
            browse_btn = QPushButton("Browse...")
            browse_btn.clicked.connect(
                lambda checked, a=alias: self.browse_poni_file(a)
            )
            h_layout.addWidget(browse_btn)

            # Clear button
            clear_btn = QPushButton("Clear")
            clear_btn.clicked.connect(lambda checked, a=alias: self.clear_poni_file(a))
            h_layout.addWidget(clear_btn)

            form_layout.addRow(f"{alias}:", h_layout)

        layout.addLayout(form_layout)

        # Buttons
        button_layout = QHBoxLayout()

        ok_btn = QPushButton("OK")
        ok_btn.clicked.connect(self.accept)
        button_layout.addWidget(ok_btn)

        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        button_layout.addWidget(cancel_btn)

        layout.addStretch()
        layout.addLayout(button_layout)

    def browse_poni_file(self, alias):
        """Open file dialog to select PONI file for the given alias."""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            f"Select PONI File for {alias}",
            "",
            "PONI Files (*.poni);;All Files (*)",
        )
        if file_path:
            self.line_edits[alias].setText(file_path)
            self.poni_files[alias] = file_path

    def clear_poni_file(self, alias):
        """Clear the PONI file selection for the given alias."""
        self.line_edits[alias].setText("")
        if alias in self.poni_files:
            del self.poni_files[alias]

    def get_poni_files(self):
        """Return dictionary of alias -> poni_file_path."""
        result = {}
        for alias, line_edit in self.line_edits.items():
            path = line_edit.text().strip()
            if path:
                result[alias] = path
        return result


class TechnicalMeasurementsMixin(_ZoneMeasurementsMixin):

    NO_SELECTION_LABEL = "— Select —"

    # Types that can be assigned to a technical measurement file in the UI.
    # NOTE: "SPECIAL" is optional and should not be required for completeness checks.
    TYPE_OPTIONS = ["AGBH", "DARK", "EMPTY", "BACKGROUND", "SPECIAL"]

    # Types required to generate a complete technical_meta_*.json (per alias).
    REQUIRED_TYPE_OPTIONS = ["AGBH", "DARK", "EMPTY", "BACKGROUND"]

    def _log_technical_event(self, message: str):
        """Log technical measurement events to the Zone Measurements log window."""
        try:
            # Use the inherited logging method from ZoneMeasurementsUIMixin
            self._append_measurement_log(f"[Technical] {message}")
        except Exception:
            # Fallback to print if logging fails
            print(f"[Technical] {message}")

    def create_technical_panel(self):
        self.aux_counter = 0
        super().create_zone_measurements()

        # Initialize continuous movement controller
        self.continuous_movement_controller = None
        self._initialize_continuous_movement_controller()

        title = "Technical Measurements"
        # Note: We don't test imports here to avoid triggering the crash at startup
        # The warning will be shown when the user first tries to use a technical feature
        self.measDock = QDockWidget(title, self)
        self.measDock.setAllowedAreas(Qt.LeftDockWidgetArea | Qt.RightDockWidgetArea)

        container = QWidget()
        outer = QVBoxLayout(container)
        outer.setContentsMargins(6, 4, 6, 4)  # Reduced margins
        outer.setSpacing(6)  # Reduced spacing between sections
        
        # Note: We don't show import warnings at startup to avoid triggering crashes
        # Warnings will be displayed when the user first tries to use technical features

        # Integration time + frames control
        it_layout = QHBoxLayout()
        it_layout.addWidget(QLabel("Integration Time (s):"))
        self.integrationTimeSpin = QDoubleSpinBox()
        # Pixet minimum integration time tested: 1 µs
        self.integrationTimeSpin.setDecimals(6)
        self.integrationTimeSpin.setRange(1e-6, 1e4)
        self.integrationTimeSpin.setSingleStep(1e-6)
        self.integrationTimeSpin.setValue(1.0)
        it_layout.addWidget(self.integrationTimeSpin)

        it_layout.addWidget(QLabel("Frames:"))
        self.captureFramesSpin = QSpinBox()
        self.captureFramesSpin.setRange(1, 1_000_000)
        self.captureFramesSpin.setValue(1)
        self.captureFramesSpin.setToolTip(
            "Capture N frames at the given integration time; frames will be averaged into a single final image"
        )
        it_layout.addWidget(self.captureFramesSpin)

        outer.addLayout(it_layout)

        # Continuous movement controls for AgBH measurements
        cm_layout = QHBoxLayout()
        self.moveContinuousCheck = QCheckBox("Move Continuous (AgBH)")
        self.moveContinuousCheck.setToolTip(
            "Enable continuous circular movement during AgBH measurements to smooth out sample inconsistencies"
        )
        cm_layout.addWidget(self.moveContinuousCheck)

        cm_layout.addWidget(QLabel("Radius (mm):"))
        self.movementRadiusSpin = QDoubleSpinBox()
        self.movementRadiusSpin.setRange(0.1, 10.0)
        self.movementRadiusSpin.setSingleStep(0.1)
        self.movementRadiusSpin.setValue(2.0)
        self.movementRadiusSpin.setDecimals(1)
        self.movementRadiusSpin.setToolTip(
            "Maximum radius for continuous movement pattern (decreases during measurement)"
        )
        cm_layout.addWidget(self.movementRadiusSpin)

        outer.addLayout(cm_layout)

        # Save folder selector
        fld = QHBoxLayout()
        fld.addWidget(QLabel("Save Folder:"))
        self.folderLE = QLineEdit()
        # Use new folder helper that respects platform defaults
        default_folder = _get_default_folder(self.config if hasattr(self, "config") else None)
        self.folderLE.setText(default_folder)

        fld.addWidget(self.folderLE, 1)
        b = QPushButton("Browse…")
        b.clicked.connect(self._browse_folder)
        fld.addWidget(b)
        outer.addLayout(fld)

        # Auxiliary Measurement controls (no label to save space)
        row = QHBoxLayout()
        self.auxBtn = QPushButton("Measure Aux")
        self.auxBtn.clicked.connect(self.measure_aux)
        row.addWidget(self.auxBtn)

        self._aux_status = QLabel("")
        row.addWidget(self._aux_status)
        self._aux_timer = QTimer(self)
        self._aux_timer.setInterval(200)
        self._aux_timer.timeout.connect(self._update_aux_status)

        self.auxNameLE = QLineEdit()
        self.auxNameLE.setPlaceholderText("Measurement name (for metadata generation)")
        self.auxNameLE.setToolTip(
            "Enter name for auxiliary measurement - used for metadata file generation"
        )
        row.addWidget(self.auxNameLE, 1)
        outer.addLayout(row)

        # Aux measurements table (compact layout for small screens)
        self.auxTable = QTableWidget()
        self.auxTable.setColumnCount(3)
        self.auxTable.installEventFilter(self)  # Delete key support
        self.auxTable.setHorizontalHeaderLabels(["File", "Type", "Alias"])

        # Make table more compact for small screens
        self.auxTable.verticalHeader().setVisible(False)  # Hide row numbers
        self.auxTable.setAlternatingRowColors(True)  # Better visual separation
        # Configure column sizing and appearance
        try:
            from PyQt5.QtGui import QFont
            from PyQt5.QtWidgets import QHeaderView

            header = self.auxTable.horizontalHeader()
            # File column takes most space, Type and Alias are compact
            header.setSectionResizeMode(0, QHeaderView.Stretch)  # File column
            header.setSectionResizeMode(1, QHeaderView.Fixed)  # Type column
            header.setSectionResizeMode(2, QHeaderView.Fixed)  # Alias column

            # Set fixed widths for Type and Alias (about 5 chars + padding)
            self.auxTable.setColumnWidth(1, 60)  # Type column
            self.auxTable.setColumnWidth(2, 60)  # Alias column

            # Optimize font and row height for small screens
            font = QFont()
            font.setPointSize(8)  # Smaller font size for more rows
            self.auxTable.setFont(font)

            # Reduce row height for more compact display
            self.auxTable.verticalHeader().setDefaultSectionSize(
                22
            )  # Smaller row height
        except Exception:
            pass
        self.auxTable.setSelectionBehavior(self.auxTable.SelectRows)
        self.auxTable.setSelectionMode(self.auxTable.ExtendedSelection)
        self.auxTable.cellDoubleClicked.connect(self._open_measurement_from_table)
        outer.addWidget(self.auxTable)

        # Compact actions layout (no groupbox to save vertical space)
        actions_layout = QHBoxLayout()
        actions_layout.setContentsMargins(0, 0, 0, 0)
        actions_layout.setSpacing(4)
        actions_layout.addWidget(QLabel("Actions:"))  # Simple label instead of groupbox

        load_btn = QPushButton("Load Files…")
        load_btn.setToolTip("Load existing technical measurement files into the table")
        load_btn.clicked.connect(self.load_technical_files)
        actions_layout.addWidget(load_btn)

        pyfai_btn = QPushButton("PyFAI")
        pyfai_btn.setToolTip("Run pyfai-calib2 in this folder")
        pyfai_btn.clicked.connect(self.run_pyfai)
        actions_layout.addWidget(pyfai_btn)

        gen_btn = QPushButton("Gen H5")
        gen_btn.setToolTip("Generate technical_<id>.h5 HDF5 container from selected rows")
        gen_btn.clicked.connect(self.generate_technical_h5)
        actions_layout.addWidget(gen_btn)

        outer.addLayout(actions_layout)

        # Real-time controls
        rt_layout = QHBoxLayout()
        rt_layout.addWidget(QLabel("Frames/⟳:"))
        self.framesSpin = QSpinBox()
        self.framesSpin.setRange(1, 1_000_000)
        self.framesSpin.setValue(1)
        rt_layout.addWidget(self.framesSpin)

        self.rtBtn = QPushButton("Real-time")
        self.rtBtn.setCheckable(True)
        self.rtBtn.clicked.connect(self._toggle_realtime)
        rt_layout.addWidget(self.rtBtn)

        outer.addLayout(rt_layout)

        # Wrap in a scroll area and add to dock
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(container)
        self.measDock.setWidget(scroll)
        self.addDockWidget(Qt.LeftDockWidgetArea, self.measDock)

        self.enable_measurement_controls(False)
        self.hardware_state_changed.connect(self.enable_measurement_controls)
        # Refresh alias models when hardware state changes
        self.hardware_state_changed.connect(
            lambda _: self.refresh_aux_table_alias_models()
        )
        # Reinitialize continuous movement controller when hardware changes
        self.hardware_state_changed.connect(
            lambda _: self._initialize_continuous_movement_controller()
        )

    def enable_measurement_controls(self, enable: bool):
        status = "enabled" if enable else "disabled"
        self._log_technical_event(f"Technical measurement controls {status}")
        widgets = [
            self.integrationTimeSpin,
            self.captureFramesSpin,
            self.moveContinuousCheck,
            self.movementRadiusSpin,
            self.folderLE,
            self.auxBtn,
            self.auxNameLE,
            self.auxTable,
            self.framesSpin,
            self.rtBtn,
        ]
        for w in widgets:
            w.setEnabled(enable)

    def _initialize_continuous_movement_controller(self):
        """Initialize the continuous movement controller if stage is available."""
        try:
            from hardware.difra.gui.technical.continuous_movement import (
                ContinuousMovementController,
            )

            # Get stage controller from hardware controller if available
            stage_controller = None
            if hasattr(self, "hardware_controller") and self.hardware_controller:
                stage_controller = self.hardware_controller.stage_controller
            elif hasattr(self, "stage_controller"):
                stage_controller = self.stage_controller

            if stage_controller:
                self.continuous_movement_controller = ContinuousMovementController(
                    stage_controller=stage_controller, parent=self
                )
                # Connect signals for monitoring
                self.continuous_movement_controller.movement_error.connect(
                    lambda msg: self._log_technical_event(f"Movement error: {msg}")
                )
                self._log_technical_event("Continuous movement controller initialized")
                logger.info("Continuous movement controller initialized")
            else:
                self._log_technical_event(
                    "No stage controller available for continuous movement"
                )
                logger.debug("No stage controller available for continuous movement")
        except ImportError as e:
            logger.warning(f"Failed to import continuous movement controller: {e}")
        except Exception as e:
            logger.error(f"Error initializing continuous movement controller: {e}", exc_info=True)

    def _browse_folder(self):
        f = QFileDialog.getExistingDirectory(self, "Select Folder")
        if f:
            self.folderLE.setText(f)

    def _start_capture(self, typ: str):
        if not _get_technical_imports():
            error_msg = (
                f"Cannot start {typ} capture - technical measurement modules failed to import. "
                "Check application logs for detailed error information. "
                "Common causes: missing pyFAI or fabio dependencies."
            )
            self._log_technical_event(error_msg)
            logger.error(error_msg)
            QMessageBox.warning(
                self,
                "Import Error",
                error_msg + "\n\nPlease check the application log file for detailed traceback."
            )
            return

        counter_attr = f"{typ.lower()}_counter"
        count = getattr(self, counter_attr, 0) + 1
        setattr(self, counter_attr, count)

        validate_folder = _get_technical_module("validate_folder")
        folder = validate_folder(self.folderLE.text())
        base = self._file_base(typ)
        base_with_count = f"{base}_{count:03d}"
        ts = time.strftime("%Y%m%d_%H%M%S")

        integration_time_s = float(self.integrationTimeSpin.value())
        frames = int(self.captureFramesSpin.value())

        # Keep filenames stable/readable at microsecond times and include frames.
        t_token = f"{integration_time_s:.6f}s"
        txt_filename_base = os.path.join(
            folder,
            f"{base_with_count}_{ts}_{t_token}_{frames}frames",
        )

        # Get stage controller for continuous movement
        stage_controller = None
        if hasattr(self, "hardware_controller") and self.hardware_controller:
            stage_controller = self.hardware_controller.stage_controller
        elif hasattr(self, "stage_controller"):
            stage_controller = self.stage_controller

        # Check if continuous movement should be enabled
        enable_continuous_movement = (
            getattr(self, "moveContinuousCheck", None) is not None
            and self.moveContinuousCheck.isChecked()
        )
        movement_radius = (
            self.movementRadiusSpin.value()
            if getattr(self, "movementRadiusSpin", None) is not None
            else 2.0
        )
        
        logger.debug(
            f"Starting {typ} capture: integration_time={integration_time_s}s, frames={frames}, "
            f"continuous_movement={enable_continuous_movement}, radius={movement_radius}mm"
        )

        CaptureWorker = _get_technical_module('CaptureWorker')
        worker = CaptureWorker(
            detector_controller=self.detector_controller,
            integration_time=integration_time_s,
            txt_filename_base=txt_filename_base,
            frames=frames,
            # Average frames into a single final image (post-conversion)
            naming_mode="normal",
            continuous_movement_controller=self.continuous_movement_controller,
            stage_controller=stage_controller,
            enable_continuous_movement=enable_continuous_movement,
            movement_radius=movement_radius,
        )
        thread = QThread()
        worker.moveToThread(thread)
        thread.started.connect(worker.run)  # .run() method needed in worker

        def _cleanup(success, result_files, t=typ):
            try:
                self._on_capture_done(success, result_files, t)
            except Exception as e:
                logger.error(f"Error in _on_capture_done for {t}: {e}", exc_info=True)
            finally:
                worker.deleteLater()
                thread.quit()
                thread.deleteLater()
                self._capture_workers.remove(worker)

        worker.finished.connect(_cleanup)
        thread.start()

        if not hasattr(self, "_capture_workers"):
            self._capture_workers = []
        self._capture_workers.append(worker)

    def _on_capture_done(self, success: bool, result_files: dict, typ: str):
        if not success:
            self._log_technical_event(f"{typ} capture failed")
            logger.warning(f"[{typ}] capture failed")
            self._aux_timer.stop()
            self._aux_status.setText("")
            return

        self._log_technical_event(
            f"{typ} capture successful: {len(result_files)} files"
        )
        logger.info(f"[{typ}] capture successful: {list(result_files.keys())}")
        self._aux_timer.stop()
        self._aux_status.setText("Processing...")

        # --- Set up worker
        if not _get_technical_imports():
            error_msg = "Cannot process files - technical imports not available"
            self._log_technical_event(error_msg)
            logger.error(error_msg)
            self._aux_status.setText("Import error")
            return

        self._log_technical_event("Processing measurement files...")
        MeasurementWorker = _get_technical_module("MeasurementWorker")
        # Match attenuation semantics: multiple frames averaged into one final image.
        frames = int(self.captureFramesSpin.value())
        worker = MeasurementWorker(
            filenames=result_files,
            frames=frames,
            average_frames=True,
        )
        worker.add_aux_item.connect(self._add_aux_item_to_list)
        worker.run()

    def _add_aux_item_to_list(self, alias, npy_path):
        """Add a new row to the Aux table with file, type and alias selectors.
        Also validates filename format: name_timestamp_..._ALIAS.ext (timestamp before alias).
        """
        from pathlib import Path

        from PyQt5.QtCore import Qt

        # Validate naming: ensure timestamp before alias
        try:
            if not self._validate_timestamp_before_alias(npy_path):
                from PyQt5.QtWidgets import QMessageBox

                QMessageBox.warning(
                    self,
                    "Filename format",
                    "File name should include timestamp before detector alias\n"
                    "Expected pattern like: name_YYYYMMDD_HHMMSS_..._ALIAS.ext",
                )
        except Exception:
            pass

        row = self.auxTable.rowCount()
        self.auxTable.insertRow(row)

        # File column (read-only, store full path in UserRole)
        display = f"{alias}: {Path(npy_path).name}"
        file_item = QTableWidgetItem(display)
        file_item.setFlags(Qt.ItemIsSelectable | Qt.ItemIsEnabled)
        file_item.setData(Qt.UserRole, str(npy_path))
        self.auxTable.setItem(row, 0, file_item)

        # Type combobox (blank by default)
        type_cb = self._make_type_combobox()
        self.auxTable.setCellWidget(row, 1, type_cb)

        # Alias combobox (preselect the source alias)
        alias_cb = self._make_alias_combobox(preselect=alias)
        self.auxTable.setCellWidget(row, 2, alias_cb)

        # Auto-set type if we can infer it from filename
        try:
            inferred_type = self._infer_type_from_filename(npy_path)
            if inferred_type:
                type_cb = self.auxTable.cellWidget(row, 1)
                if type_cb and hasattr(type_cb, "findText"):
                    idx = type_cb.findText(inferred_type)
                    if idx >= 0:
                        type_cb.setCurrentIndex(idx)
                        self._log_technical_event(
                            f"Added {inferred_type} measurement: {alias} ({os.path.basename(npy_path)})"
                        )
        except Exception:
            pass

    def _file_base(self, typ: str) -> str:
        le: QLineEdit = getattr(self, f"{typ.lower()}NameLE")
        txt = le.text().strip().replace(" ", "_")
        return txt or typ.lower()

    # ---- Upload and validation helpers ----
    def _validate_timestamp_before_alias(self, file_path: str) -> bool:
        """Return True if the file name has a timestamp (YYYYMMDD_HHMMSS) before alias.
        Accepts names like: name_YYYYMMDD_HHMMSS_..._ALIAS.ext"""
        base = os.path.basename(file_path)
        # Remove extension
        name, _ext = os.path.splitext(base)
        # Expect at least 3 tokens separated by underscores
        toks = name.split("_")
        if len(toks) < 3:
            return False
        # Look for timestamp token 'YYYYMMDD_HHMMSS' across two tokens or combined with underscore
        # Our generator uses a single token with embedded '_': YYYYMMDD_HHMMSS
        stamp_match = re.search(r"\d{8}_\d{6}", name)
        if not stamp_match:
            return False
        # Ensure alias is the last token
        alias = toks[-1]
        # Minimal alias check: alphanumeric
        if not re.fullmatch(r"[A-Za-z0-9]+", alias):
            return False
        # Ensure the timestamp appears before the alias
        return stamp_match.start() < (len(name) - len(alias))

    def _infer_alias_from_filename(self, file_path: str) -> str:
        base = os.path.basename(file_path)
        alias = os.path.splitext(base)[0].split("_")[-1]
        # Validate against active aliases if available
        try:
            active_aliases = self._get_active_detector_aliases()
            if alias in active_aliases:
                return alias
        except Exception:
            pass
        return alias  # fallback

    def _infer_type_from_filename(self, file_path: str) -> str:
        """Infer measurement type from filename patterns."""
        base = os.path.basename(file_path).lower()
        # Check for explicit type tokens first
        for type_option in self.TYPE_OPTIONS:
            if type_option.lower() in base:
                return type_option

        # Variations / legacy naming
        if "background" in base:
            return "BACKGROUND"
        if "dark" in base:
            return "DARK"
        if "empty" in base:
            return "EMPTY"
        if "agbh" in base:
            return "AGBH"
        return None  # No match found

    def load_technical_files(self):
        """Load existing technical measurement files into the aux table.
        Validates file naming and tries to infer alias from the filename."""
        from PyQt5.QtWidgets import QFileDialog, QMessageBox

        files, _ = QFileDialog.getOpenFileNames(
            self,
            "Load Technical Measurement Files",
            str(self.folderLE.text() or ""),
            "NumPy Arrays (*.npy);;Text Files (*.txt);;All Files (*)",
        )
        if not files:
            return

        self._log_technical_event(f"Loading {len(files)} technical files...")

        for fpath in files:
            # Convert .txt to .npy next to it (non-destructive)
            path_to_use = fpath
            try:
                if fpath.lower().endswith(".txt"):
                    data = np.loadtxt(fpath)
                    npy_path = os.path.splitext(fpath)[0] + ".npy"
                    np.save(npy_path, data)
                    path_to_use = npy_path
            except Exception as e:
                QMessageBox.warning(
                    self,
                    "Conversion failed",
                    f"Failed to convert TXT to NPY for:\n{fpath}\nError: {e}",
                )
                continue

            # Validate filename format
            if not self._validate_timestamp_before_alias(path_to_use):
                QMessageBox.warning(
                    self,
                    "Filename format",
                    "File name should include timestamp before detector alias\n"
                    "Expected pattern like: name_YYYYMMDD_HHMMSS_..._ALIAS.ext",
                )
                # Continue adding anyway, but user is informed

            alias = self._infer_alias_from_filename(path_to_use)
            self._add_aux_item_to_list(alias, path_to_use)

            # Auto-set type if we can infer it from filename
            try:
                inferred_type = self._infer_type_from_filename(path_to_use)
                if inferred_type:
                    row_idx = self.auxTable.rowCount() - 1
                    type_cb = self.auxTable.cellWidget(row_idx, 1)
                    if type_cb and hasattr(type_cb, "findText"):
                        idx = type_cb.findText(inferred_type)
                        if idx >= 0:
                            type_cb.setCurrentIndex(idx)
            except Exception:
                pass

    # ---- Persist/restore aux table in global state ----
    def build_aux_state(self):
        """Serialize current auxTable rows to a list for state saving."""
        rows = []
        try:
            if not hasattr(self, "auxTable") or self.auxTable is None:
                return rows
            for r in range(self.auxTable.rowCount()):
                file_item = self.auxTable.item(r, 0)
                file_path = (
                    file_item.data(Qt.UserRole) if file_item is not None else None
                )
                # Type
                type_cb = self.auxTable.cellWidget(r, 1)
                type_text = None
                try:
                    if type_cb is not None:
                        t = type_cb.currentText()
                        if t and t != self.NO_SELECTION_LABEL:
                            type_text = t
                except Exception:
                    pass
                # Alias
                alias_cb = self.auxTable.cellWidget(r, 2)
                alias_text = None
                try:
                    if alias_cb is not None:
                        a = alias_cb.currentText()
                        if a and a != self.NO_SELECTION_LABEL:
                            alias_text = a
                except Exception:
                    pass
                rows.append(
                    {"file_path": file_path, "type": type_text, "alias": alias_text}
                )
        except Exception as e:
            print(f"Error building aux state: {e}")
        return rows

    def restore_technical_aux_rows(self, rows):
        """Restore auxTable rows from previously saved state."""
        try:
            if not hasattr(self, "auxTable") or self.auxTable is None:
                return
            # Clear existing rows
            self.auxTable.setRowCount(0)
            for row in rows or []:
                fpath = row.get("file_path")
                alias = row.get("alias") or self._infer_alias_from_filename(fpath or "")
                self._add_aux_item_to_list(alias or "", fpath or "")
                # Set type if provided
                try:
                    rix = self.auxTable.rowCount() - 1
                    type_cb = self.auxTable.cellWidget(rix, 1)
                    if type_cb is not None and row.get("type"):
                        idx = (
                            type_cb.findText(row["type"])
                            if hasattr(type_cb, "findText")
                            else -1
                        )
                        if idx >= 0:
                            type_cb.setCurrentIndex(idx)
                except Exception:
                    pass
        except Exception as e:
            print(f"Error restoring aux rows: {e}")

    def measure_aux(self):
        # Check if technical imports are available before starting
        if not _get_technical_imports():
            self._log_technical_event("Cannot start Aux measurement - technical imports not available")
            print("❌ Cannot start Aux measurement - technical measurements disabled due to import errors")
            from PyQt5.QtWidgets import QMessageBox
            QMessageBox.warning(
                self,
                "Technical Measurements Unavailable",
                "Technical measurements are disabled due to import errors.\n\nCheck the console for details."
            )
            return
        
        # Check for existing HDF5 containers before starting new measurements
        folder = (self.folderLE.text() or "").strip()
        if folder and os.path.isdir(folder):
            try:
                from hardware.difra.utils.technical_h5_archival import (
                    TechnicalH5Archival,
                    format_archival_summary,
                )
                
                containers = TechnicalH5Archival.find_h5_containers(folder)
                if containers:
                    # Found existing containers - prompt user
                    container_list = "\n".join([f"  • {c.name}" for c in containers[:5]])
                    if len(containers) > 5:
                        container_list += f"\n  ... and {len(containers) - 5} more"
                    
                    message = (
                        f"Found {len(containers)} existing HDF5 container(s) in:\n"
                        f"{folder}\n\n"
                        f"{container_list}\n\n"
                        f"These will be moved to '{TechnicalH5Archival.STORAGE_SUBFOLDER}' "
                        f"folder and associated .npy files will be cleaned up.\n\n"
                        f"Do you want to archive them before starting new measurements?"
                    )
                    
                    reply = QMessageBox.question(
                        self,
                        "Archive Existing Containers?",
                        message,
                        QMessageBox.Yes | QMessageBox.No | QMessageBox.Cancel,
                        QMessageBox.Yes,
                    )
                    
                    if reply == QMessageBox.Cancel:
                        self._log_technical_event("Aux measurement cancelled by user")
                        return
                    elif reply == QMessageBox.Yes:
                        # Archive containers and clean up files
                        self._log_technical_event(
                            f"Archiving {len(containers)} HDF5 container(s)..."
                        )
                        
                        # Get active detector aliases for cleanup
                        try:
                            aliases = self._get_active_detector_aliases()
                        except Exception:
                            aliases = ["PRIMARY", "SECONDARY"]  # Fallback
                        
                        measurement_types = ["DARK", "EMPTY", "BACKGROUND", "AGBH", "WATER", "SPECIAL"]
                        
                        archived, cleaned, errors = TechnicalH5Archival.archive_all_and_cleanup(
                            folder,
                            measurement_types=measurement_types,
                            aliases=aliases,
                            add_timestamp=True,
                        )
                        
                        summary = format_archival_summary(archived, cleaned, errors)
                        self._log_technical_event(f"Archival complete: {archived} archived, {cleaned} cleaned")
                        
                        QMessageBox.information(
                            self,
                            "Archival Complete",
                            f"Archival Summary:\n\n{summary}",
                        )
                    else:  # QMessageBox.No
                        self._log_technical_event("User chose to skip archival")
                        logger.info("User skipped HDF5 container archival")
                        
            except Exception as e:
                logger.error(f"Error checking for existing containers: {e}", exc_info=True)
                # Non-fatal - continue with measurement
                self._log_technical_event(f"Warning: Failed to check for existing containers: {e}")
        
        self._log_technical_event("Starting auxiliary measurement...")
        self._aux_start = time.time()
        self._aux_spinner_state = 0
        self._aux_status.setText("0 s ⁑")
        self._aux_timer.start()
        self._start_capture("Aux")

    def _open_measurement_from_table(self, row: int, _col: int):
        """Open measurement window for the selected row."""
        from PyQt5.QtCore import Qt

        file_item = self.auxTable.item(row, 0)
        if not file_item:
            return
        file_path = file_item.data(Qt.UserRole)

        self._log_technical_event(
            f"Opening measurement file: {os.path.basename(file_path) if file_path else 'Unknown'}"
        )

        # Prefer alias from Alias #1 if selected, else try to infer from display text
        alias_cb = self.auxTable.cellWidget(row, 2)
        alias = None
        if isinstance(alias_cb, QComboBox):
            a = alias_cb.currentText().strip()
            if a and a != self.NO_SELECTION_LABEL:
                alias = a

        if not alias:
            disp = file_item.text()
            if ":" in disp:
                alias = disp.split(":", 1)[0].strip()

        if not alias:
            # Fallback to first controller alias
            try:
                alias = next(iter(self.detector_controller))
            except Exception:
                alias = None

        if not _get_technical_imports():
            self._log_technical_event("Cannot open measurement window - technical imports not available")
            return
            
        show_measurement_window = _get_technical_module('show_measurement_window')
        show_measurement_window(
            file_path, self.masks.get(alias), self.ponis.get(alias), self
        )

    def run_pyfai(self):
        self._log_technical_event("Starting PyFAI calibration...")
        env = self.config.get("conda")
        if not env:
            self._log_technical_event("Error: No conda environment configured")
            print("❌ No conda env set in self.config['conda']")
            return

        validate_folder = _get_technical_module('validate_folder')
        folder = validate_folder(self.folderLE.text())

        if os.name == "nt":
            cmd = (
                f"CALL conda activate {env} " f'&& cd /d "{folder}" ' f"&& pyfai-calib2"
            )
            start_cmd = f'start cmd /K "{cmd}"'
            try:
                subprocess.Popen(start_cmd, shell=True)
                self._log_technical_event("PyFAI calibration launched in new window")
                print("▶️ Launched PyFai in new cmd window.")
            except Exception as e:
                self._log_technical_event(f"Failed to launch PyFAI on Windows: {e}")
                print("❌ Failed to launch PyFai on Windows:", e)
        else:
            # Use conda run instead of conda activate for better compatibility
            try:
                # Try to open in a new terminal window (macOS)
                if sys.platform == 'darwin':
                    # macOS: Create a temporary shell script and open it with Terminal
                    # This avoids needing AppleScript permissions
                    import tempfile
                    
                    script_content = f'''#!/bin/bash
cd "{folder}"
echo "Starting PyFAI calibration in conda environment: {env}"
echo "Folder: {folder}"
echo ""
conda run -n {env} pyfai-calib2
if [ $? -ne 0 ]; then
    echo ""
    echo "Error: Failed to launch PyFAI. Check that:"
    echo "  1. Conda environment '{env}' exists (run: conda env list)"
    echo "  2. pyfai-calib2 is installed (run: conda run -n {env} which pyfai-calib2)"
    echo ""
    echo "Press any key to close..."
    read -n 1
fi
'''
                    
                    # Create temporary script file
                    with tempfile.NamedTemporaryFile(mode='w', suffix='.command', delete=False) as f:
                        f.write(script_content)
                        script_path = f.name
                    
                    # Make it executable
                    import os as os_module
                    os_module.chmod(script_path, 0o755)
                    
                    # Open with Terminal using 'open' command (doesn't require permissions)
                    subprocess.Popen(['open', '-a', 'Terminal', script_path])
                    self._log_technical_event(f"PyFAI calibration script created: {script_path}")
                else:
                    # Linux: try common terminal emulators
                    bash_cmd = (
                        f'cd "{folder}" && '
                        f'echo "Starting PyFAI in environment: {env}" && '
                        f'conda run -n {env} pyfai-calib2 || '
                        f'(echo "\\nError: Failed to launch PyFAI"; read -p "Press Enter to close...")'
                    )
                    for terminal in ['gnome-terminal', 'konsole', 'xterm']:
                        try:
                            subprocess.Popen([terminal, '--', 'bash', '-c', bash_cmd])
                            break
                        except FileNotFoundError:
                            continue
                self._log_technical_event(
                    "PyFAI calibration launched in new terminal window"
                )
                print("▶️ Launched PyFai in new terminal window.")
            except Exception as e:
                self._log_technical_event(f"Failed to launch PyFAI on Unix: {e}")
                print("❌ Failed to launch PyFai on Unix:", e)

    def initialize_hardware(self):
        pass

    def _update_aux_status(self):
        elapsed = int(time.time() - self._aux_start)
        spinner = ["⁑", "⁙", "⁹", "⁸", "‼", "‴", "…", "‧", " ", "‏"]
        ch = spinner[self._aux_spinner_state % len(spinner)]
        self._aux_spinner_state += 1
        self._aux_status.setText(f"{elapsed} s {ch}")

        # Log every 10 seconds
        if (
            elapsed > 0
            and elapsed % 10 == 0
            and self._aux_spinner_state % len(spinner) == 0
        ):
            self._log_technical_event(
                f"Auxiliary measurement in progress: {elapsed} seconds"
            )

    def _toggle_realtime(self, checked: bool):
        if checked:
            self._log_technical_event("Starting real-time measurement display")
            self._start_realtime()
            self.rtBtn.setText("Stop RT")
        else:
            self._log_technical_event("Stopping real-time measurement display")
            self._stop_realtime()
            self.rtBtn.setText("Real-time")

    # ---- Deletion of selected Aux rows via Delete key (no file removal) ----
    def delete_selected_aux_rows(self):
        try:
            if not hasattr(self, "auxTable") or self.auxTable is None:
                return
            sel_model = self.auxTable.selectionModel()
            if not sel_model:
                return
            rows = sorted({ix.row() for ix in sel_model.selectedRows()}, reverse=True)
            if not rows:
                return
            for r in rows:
                try:
                    self.auxTable.removeRow(r)
                except Exception:
                    pass
        except Exception as e:
            print(f"Error deleting selected aux rows: {e}")

    def eventFilter(self, source, event):
        # Handle Delete key for auxTable to remove rows only from UI/state
        if (
            source is getattr(self, "auxTable", None)
            and event.type() == QEvent.KeyPress
        ):
            try:
                if event.key() == Qt.Key_Delete:
                    self.delete_selected_aux_rows()
                    return True
            except Exception:
                pass
        # Chain to super to allow other mixins (e.g., ZonePoints) to handle their filters
        return super().eventFilter(source, event)

    def _start_realtime(self):
        exposure = float(self.integrationTimeSpin.value())
        self._rt_queue = queue.Queue()

        plt.ion()
        detector_aliases = list(self.detector_controller.keys())
        n_det = len(detector_aliases)
        self._rt_img = {}
        self._rt_last_frame = {}  # <--- Cache for latest frame per alias

        # One subplot per detector alias
        fig, axes = plt.subplots(1, n_det, figsize=(5 * n_det, 5))
        if n_det == 1:
            axes = [axes]

        for ax, alias in zip(axes, detector_aliases):
            size = getattr(self.detector_controller[alias], "size", (256, 256))
            self._rt_img[alias] = ax.imshow(
                np.zeros(size), origin="lower", interpolation="none"
            )
            ax.set_title(alias)
        self._rt_fig = fig
        plt.show()

        self._plot_timer = QTimer(self)
        self._plot_timer.setInterval(50)
        self._plot_timer.timeout.connect(self._rt_plot_tick)
        self._plot_timer.start()

        def callback(frames_dict):
            # Cache most recent frame per alias
            for alias, frame in frames_dict.items():
                self._rt_last_frame[alias] = frame
            self._rt_queue.put(True)  # Just a signal to the timer

        # Start stream on all detectors
        for controller in self.detector_controller.values():
            controller.start_stream(
                callback=callback, exposure=exposure, interval=0.0, frames=1
            )

    def _rt_plot_tick(self):
        # Drain the queue (we only need to plot once per timer tick)
        while True:
            try:
                _ = self._rt_queue.get_nowait()
            except queue.Empty:
                break
        # Update all subplots with their latest frame
        for alias in self._rt_img:
            frame = self._rt_last_frame.get(alias)
            if frame is not None:
                self._rt_img[alias].set_data(frame)
                self._rt_img[alias].set_clim(frame.min(), frame.max())
        self._rt_fig.canvas.draw_idle()

    def _stop_realtime(self):
        for controller in self.detector_controller.values():
            controller.stop_stream()
        if hasattr(self, "_plot_timer"):
            self._plot_timer.stop()
            del self._plot_timer
        import matplotlib.pyplot as plt

        plt.close(self._rt_fig)
        del self._rt_queue
        del self._rt_last_frame

    # -------------------- Helpers for Aux Table --------------------
    def _get_active_detector_aliases(self):
        """Return aliases from settings (main.json), honoring DEV/dev_active_detectors.
        This intentionally reads from config instead of live hardware."""
        dev_mode = self.config.get("DEV", False)
        ids = (
            self.config.get("dev_active_detectors", [])
            if dev_mode
            else self.config.get("active_detectors", [])
        )
        return [
            d.get("alias")
            for d in self.config.get("detectors", [])
            if d.get("id") in ids
        ]

    def _get_active_detector_ids(self):
        """Return active detector IDs from config (main.json), honoring DEV/dev_active_detectors."""
        dev_mode = self.config.get("DEV", False)
        return (
            self.config.get("dev_active_detectors", [])
            if dev_mode
            else self.config.get("active_detectors", [])
        )

    def _parse_poni_distance_m(self, poni_text: str):
        """Parse Distance from PONI text in meters. Returns None if not found/invalid."""
        if not poni_text:
            return None
        try:
            m = re.search(r"^Distance:\s*([0-9.eE+-]+)", poni_text, flags=re.MULTILINE)
            return float(m.group(1)) if m else None
        except Exception:
            return None

    def _prompt_distance_cm(self, default_cm: float = None):
        """Prompt user for sample-detector distance in cm. Returns None if canceled."""
        default_val = 17.0 if default_cm is None else float(default_cm)
        dist_cm, ok = QInputDialog.getDouble(
            self,
            "Sample-Detector Distance",
            "Enter sample-detector distance (cm):",
            default_val,
            0.01,
            100000.0,
            3,
        )
        return float(dist_cm) if ok else None

    def _normalize_technical_type(self, typ: str) -> str:
        """Normalize UI type labels to schema technical types."""
        if typ == "SPECIAL":
            return "WATER"
        return typ

    def _make_type_combobox(self):
        cb = QComboBox()
        cb.addItem(self.NO_SELECTION_LABEL, None)
        for t in self.TYPE_OPTIONS:
            cb.addItem(t, t)
        return cb

    def _make_alias_combobox(self, preselect=None):
        cb = QComboBox()
        cb.addItem(self.NO_SELECTION_LABEL, None)
        for alias in self._get_active_detector_aliases():
            cb.addItem(alias, alias)
        if preselect:
            idx = cb.findText(preselect)
            if idx >= 0:
                cb.setCurrentIndex(idx)
        return cb

    def refresh_aux_table_alias_models(self):
        aliases = self._get_active_detector_aliases()
        for row in range(self.auxTable.rowCount()):
            cb = self.auxTable.cellWidget(row, 2)
            if not isinstance(cb, QComboBox):
                continue
            current = cb.currentText()
            cb.blockSignals(True)
            cb.clear()
            cb.addItem(self.NO_SELECTION_LABEL, None)
            for a in aliases:
                cb.addItem(a, a)
            # restore selection if still present
            if current and current in aliases:
                cb.setCurrentText(current)
            cb.blockSignals(False)

    # -------------------- Generate Technical Meta --------------------
    def generate_technical_meta(self):
        from pathlib import Path

        self._log_technical_event("Generating technical metadata...")

        # Validate selection
        sel = (
            self.auxTable.selectionModel().selectedRows()
            if self.auxTable.selectionModel()
            else []
        )
        rows = [idx.row() for idx in sel]
        if not rows:
            self._log_technical_event("Error: No rows selected for metadata generation")
            QMessageBox.warning(
                self, "No Selection", "Select one or more rows in the Aux table."
            )
            return

        # Validate name and folder
        name = (self.auxNameLE.text() or "").strip()
        if not name:
            QMessageBox.warning(
                self, "Missing Name", "Enter a name in the Aux Measurement field."
            )
            return
        safe_name = name.replace(" ", "_")
        # Use the explicit folder path for meta generation.
        # (Do not auto-fallback to CWD; if the folder is invalid/unwritable we should stop.)
        folder = (self.folderLE.text() or "").strip()
        if not folder or not os.path.isdir(folder):
            QMessageBox.warning(self, "Invalid Folder", "Select a valid save folder.")
            return
        if not os.access(folder, os.W_OK):
            QMessageBox.warning(
                self,
                "Folder Not Writable",
                "Selected save folder is not writable. Choose a different folder.",
            )
            return

        out_path = os.path.join(folder, f"technical_meta_{safe_name}.json")
        if os.path.exists(out_path):
            res = QMessageBox.question(
                self,
                "Overwrite?",
                f"File exists:\n{out_path}\n\nDo you want to overwrite it?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            )
            if res != QMessageBox.Yes:
                return

        meta = {}
        seen_pairs = set()  # (type, alias)

        # Get active detector aliases for validation
        try:
            active_aliases = self._get_active_detector_aliases()
        except Exception:
            active_aliases = []

        for row in rows:
            file_item = self.auxTable.item(row, 0)
            if not file_item:
                continue
            file_path = file_item.data(Qt.UserRole)
            if not file_path or not os.path.exists(file_path):
                QMessageBox.warning(
                    self, "Missing File", f"Row {row+1}: file path does not exist."
                )
                return

            # Type
            type_cb = self.auxTable.cellWidget(row, 1)
            if (
                not isinstance(type_cb, QComboBox)
                or type_cb.currentText() == self.NO_SELECTION_LABEL
            ):
                QMessageBox.warning(
                    self, "Missing Type", f"Row {row+1}: select measurement type."
                )
                return
            typ = type_cb.currentText()

            # Alias (must be selected)
            cb = self.auxTable.cellWidget(row, 2)
            if (
                not isinstance(cb, QComboBox)
                or cb.currentText() == self.NO_SELECTION_LABEL
            ):
                QMessageBox.warning(
                    self, "Missing Alias", f"Row {row+1}: select an alias."
                )
                return
            al = cb.currentText()

            base = os.path.basename(file_path)
            dst = meta.setdefault(typ, {})
            pair = (typ, al)
            if pair in seen_pairs or al in dst:
                QMessageBox.warning(
                    self,
                    "Duplicate Assignment",
                    f"Measurement for type '{typ}' and alias '{al}' is already assigned.",
                )
                return
            dst[al] = base
            seen_pairs.add(pair)

        # Enforce completeness: all REQUIRED measurement types must be present, and for each alias
        required_types = set(
            getattr(self, "REQUIRED_TYPE_OPTIONS", None)
            or ["AGBH", "DARK", "EMPTY", "BACKGROUND"]
        )

        # 1) Ensure at least one row selected for each required type
        types_in_meta = {t for t in meta.keys() if t in required_types}
        missing_types = sorted(required_types - types_in_meta)
        if missing_types:
            QMessageBox.warning(
                self,
                "Missing Measurement Types",
                "The following measurement types are missing from your selection:\n\n"
                + ", ".join(missing_types)
                + "\n\nPlease include at least one measurement for each required type before generating the meta file.",
            )
            return

        # 2) Ensure per-alias coverage for each required type
        # Prefer active aliases from config; if unavailable, fall back to aliases seen in selection
        aliases_in_selection = set()
        for type_map in meta.values():
            if isinstance(type_map, dict):
                aliases_in_selection.update(type_map.keys())
        aliases_to_check = active_aliases or sorted(aliases_in_selection)

        missing_pairs = []
        for t in sorted(required_types):
            type_map = meta.get(t, {})
            for a in aliases_to_check:
                if a not in type_map:
                    missing_pairs.append(f"{t} → {a}")

        if missing_pairs:
            QMessageBox.warning(
                self,
                "Incomplete Technical Set",
                "All measurement types must be provided for each detector alias.\n\nMissing combinations:\n"
                + "\n".join(missing_pairs),
            )
            return

        # Get unique aliases from selected measurements for PONI file selection
        unique_aliases = set()
        for row in rows:
            cb = self.auxTable.cellWidget(row, 2)
            if (
                isinstance(cb, QComboBox)
                and cb.currentText() != self.NO_SELECTION_LABEL
            ):
                unique_aliases.add(cb.currentText())

        # Show PONI file selection dialog if we have aliases
        poni_lab = {}
        if unique_aliases:
            # Get current PONI files if available
            current_poni_files = getattr(self, "poni_files", {})

            poni_dialog = PoniFileSelectionDialog(
                aliases=sorted(unique_aliases),
                current_poni_files=current_poni_files,
                parent=self,
            )

            if poni_dialog.exec_() == QDialog.Accepted:
                selected_poni_files = poni_dialog.get_poni_files()
                poni_lab_path = {}
                poni_lab_values = {}

                # Process each selected PONI file
                for alias, file_path in selected_poni_files.items():
                    # Store filename for PONI_LAB
                    poni_lab[alias] = os.path.basename(file_path)

                    # Store full path for PONI_LAB_PATH
                    poni_lab_path[alias] = file_path

                    # Read and store PONI file content for PONI_LAB_VALUES
                    try:
                        with open(file_path, "r", encoding="utf-8") as f:
                            poni_content = f.read()
                            poni_lab_values[alias] = poni_content
                    except Exception as e:
                        QMessageBox.warning(
                            self,
                            "PONI File Read Error",
                            f"Failed to read PONI file for {alias}:\n{file_path}\n\nError: {e}\n\nContinuing without this PONI file content.",
                        )
                        # Still include the filename and path, but mark content as unavailable
                        poni_lab_values[alias] = (
                            f"# ERROR: Could not read PONI file content\n# File: {file_path}\n# Error: {str(e)}"
                        )

                # Store additional PONI data for later use
                self._temp_poni_lab_path = poni_lab_path
                self._temp_poni_lab_values = poni_lab_values
            else:
                # User cancelled PONI selection, ask if they want to continue without PONI files
                res = QMessageBox.question(
                    self,
                    "No PONI Files Selected",
                    "Do you want to generate the technical meta file without PONI calibration files?",
                    QMessageBox.Yes | QMessageBox.No,
                    QMessageBox.No,
                )
                if res != QMessageBox.Yes:
                    return

        # Add PONI sections to meta if any PONI files were selected
        if poni_lab:
            meta["PONI_LAB"] = poni_lab

        # Add PONI_LAB_PATH section if available
        if hasattr(self, "_temp_poni_lab_path") and self._temp_poni_lab_path:
            meta["PONI_LAB_PATH"] = self._temp_poni_lab_path

        # Add PONI_LAB_VALUES section if available
        if hasattr(self, "_temp_poni_lab_values") and self._temp_poni_lab_values:
            meta["PONI_LAB_VALUES"] = self._temp_poni_lab_values

        # Add or reuse a calibration group hash so multiple files can be grouped together
        try:
            group_hash = getattr(self, "calibration_group_hash", None)
            if not group_hash:
                group_hash = uuid.uuid4().hex[:16]
                setattr(self, "calibration_group_hash", group_hash)
            meta["CALIBRATION_GROUP_HASH"] = group_hash
        except Exception:
            # Non-fatal; proceed without the group hash if something unexpected happens
            pass

        # Write JSON
        try:
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(meta, f, indent=2)
        except Exception as e:
            QMessageBox.critical(
                self, "Write Error", f"Failed to write meta file:\n{e}"
            )
            return
        
        # Clean up temporary PONI data variables
        if hasattr(self, "_temp_poni_lab_path"):
            delattr(self, "_temp_poni_lab_path")
        if hasattr(self, "_temp_poni_lab_values"):
            delattr(self, "_temp_poni_lab_values")

        # Summary
        try:
            summary_lines = []
            for k, v in meta.items():
                if isinstance(v, dict):
                    summary_lines.append(f"{k}: {len(v)} file(s)")
                else:
                    summary_lines.append(f"{k}: {v}")
            summary = "\n".join(summary_lines) or "(empty)"
        except Exception:
            summary = "(summary unavailable)"
        
        self._log_technical_event(
            f"Technical metadata generated: {os.path.basename(out_path)}"
        )
        
        QMessageBox.information(
            self, "Meta Generated", f"Saved to:\n{out_path}\n\nSummary:\n{summary}"
        )
        
        # Now generate HDF5 container
        self.generate_technical_h5()

    # -------------------- Generate Technical HDF5 --------------------
    def generate_technical_h5(self):
        from hardware.difra.data.hdf5 import schema_v1, technical_container

        self._log_technical_event("Generating technical HDF5 container...")

        # Validate selection
        sel = (
            self.auxTable.selectionModel().selectedRows()
            if self.auxTable.selectionModel()
            else []
        )
        rows = [idx.row() for idx in sel]
        if not rows:
            self._log_technical_event("Error: No rows selected for HDF5 generation")
            QMessageBox.warning(
                self, "No Selection", "Select one or more rows in the Aux table."
            )
            return

        # Validate folder
        folder = (self.folderLE.text() or "").strip()
        if not folder or not os.path.isdir(folder):
            QMessageBox.warning(self, "Invalid Folder", "Select a valid save folder.")
            return
        if not os.access(folder, os.W_OK):
            QMessageBox.warning(
                self,
                "Folder Not Writable",
                "Selected save folder is not writable. Choose a different folder.",
            )
            return

        aux_measurements = {}
        seen_pairs = set()

        # Get active detector aliases for validation
        try:
            active_aliases = self._get_active_detector_aliases()
        except Exception:
            active_aliases = []

        for row in rows:
            file_item = self.auxTable.item(row, 0)
            if not file_item:
                continue
            file_path = file_item.data(Qt.UserRole)
            if not file_path or not os.path.exists(file_path):
                QMessageBox.warning(
                    self, "Missing File", f"Row {row+1}: file path does not exist."
                )
                return

            # Type
            type_cb = self.auxTable.cellWidget(row, 1)
            if (
                not isinstance(type_cb, QComboBox)
                or type_cb.currentText() == self.NO_SELECTION_LABEL
            ):
                QMessageBox.warning(
                    self, "Missing Type", f"Row {row+1}: select measurement type."
                )
                return
            typ_ui = type_cb.currentText()
            typ = self._normalize_technical_type(typ_ui)

            # Alias (must be selected)
            cb = self.auxTable.cellWidget(row, 2)
            if (
                not isinstance(cb, QComboBox)
                or cb.currentText() == self.NO_SELECTION_LABEL
            ):
                QMessageBox.warning(
                    self, "Missing Alias", f"Row {row+1}: select an alias."
                )
                return
            alias = cb.currentText()

            if typ not in schema_v1.ALL_TECHNICAL_TYPES:
                QMessageBox.warning(
                    self,
                    "Invalid Type",
                    f"Type '{typ_ui}' is not supported for HDF5.\n"
                    f"Supported: {', '.join(schema_v1.ALL_TECHNICAL_TYPES)}",
                )
                return

            if typ_ui == "SPECIAL":
                self._log_technical_event("Mapping type SPECIAL → WATER for HDF5")

            # Ensure unique (type, alias)
            pair = (typ, alias)
            if pair in seen_pairs:
                QMessageBox.warning(
                    self,
                    "Duplicate Assignment",
                    f"Measurement for type '{typ_ui}' and alias '{alias}' is already assigned.",
                )
                return

            aux_measurements.setdefault(typ, {})[alias] = file_path
            seen_pairs.add(pair)

        # Enforce completeness: all REQUIRED measurement types must be present, and for each alias
        required_types = set(
            getattr(self, "REQUIRED_TYPE_OPTIONS", None)
            or ["AGBH", "DARK", "EMPTY", "BACKGROUND"]
        )

        # 1) Ensure at least one row selected for each required type
        types_in_meta = {t for t in aux_measurements.keys() if t in required_types}
        missing_types = sorted(required_types - types_in_meta)
        if missing_types:
            QMessageBox.warning(
                self,
                "Missing Measurement Types",
                "The following measurement types are missing from your selection:\n\n"
                + ", ".join(missing_types)
                + "\n\nPlease include at least one measurement for each required type before generating HDF5.",
            )
            return

        # 2) Ensure per-alias coverage for each required type
        aliases_in_selection = set()
        for type_map in aux_measurements.values():
            if isinstance(type_map, dict):
                aliases_in_selection.update(type_map.keys())
        aliases_to_check = active_aliases or sorted(aliases_in_selection)

        missing_pairs = []
        for t in sorted(required_types):
            type_map = aux_measurements.get(t, {})
            for a in aliases_to_check:
                if a not in type_map:
                    missing_pairs.append(f"{t} → {a}")

        if missing_pairs:
            QMessageBox.warning(
                self,
                "Incomplete Technical Set",
                "All measurement types must be provided for each detector alias.\n\nMissing combinations:\n"
                + "\n".join(missing_pairs),
            )
            return

        # Collect PONI data (prefer file selection, fallback to in-memory PONI)
        pony_data = {}
        missing_pony = []
        selected_poni_files = {}

        if aliases_to_check:
            current_poni_files = getattr(self, "poni_files", {})
            poni_dialog = PoniFileSelectionDialog(
                aliases=sorted(aliases_to_check),
                current_poni_files=current_poni_files,
                parent=self,
            )

            if poni_dialog.exec_() == QDialog.Accepted:
                selected_poni_files = poni_dialog.get_poni_files() or {}
            else:
                res = QMessageBox.question(
                    self,
                    "PONI Files",
                    "Use currently loaded PONI values instead of selecting files?",
                    QMessageBox.Yes | QMessageBox.No,
                    QMessageBox.Yes,
                )
                if res != QMessageBox.Yes:
                    return

        for alias in aliases_to_check:
            poni_content = None
            poni_filename = None

            file_path = selected_poni_files.get(alias)
            if file_path and os.path.exists(file_path):
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        poni_content = f.read()
                    poni_filename = os.path.basename(file_path)
                except Exception as e:
                    QMessageBox.warning(
                        self,
                        "PONI File Read Error",
                        f"Failed to read PONI file for {alias}:\n{file_path}\n\nError: {e}\n\n"
                        "Falling back to current PONI values if available.",
                    )

            if not poni_content:
                try:
                    poni_content = (getattr(self, "ponis", {}) or {}).get(alias)
                    poni_meta = (getattr(self, "poni_files", {}) or {}).get(alias, {})
                    poni_filename = poni_meta.get("name") or f"{alias}.poni"
                except Exception:
                    poni_content = None

            if poni_content:
                pony_data[alias] = (poni_content, poni_filename or f"{alias}.poni")
            else:
                missing_pony.append(alias)

        if missing_pony:
            res = QMessageBox.question(
                self,
                "Missing PONI Data",
                "No PONI data found for:\n"
                + ", ".join(missing_pony)
                + "\n\nContinue without these PONI datasets?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            )
            if res != QMessageBox.Yes:
                return

        # Determine distance from PONI content (meters -> cm)
        distances_m = []
        poni_distance_cm = None  # Real distance from PONI file
        
        for content, _fname in pony_data.values():
            d = self._parse_poni_distance_m(content)
            if d is not None:
                distances_m.append(d)

        # Check if all PONI files have consistent distance
        if distances_m:
            ref = distances_m[0]
            poni_distance_cm = ref * 100.0  # Convert meters to cm
            
            if any(abs(d - ref) > 1e-4 for d in distances_m[1:]):
                QMessageBox.warning(
                    self,
                    "Distance Mismatch Between PONIs",
                    f"Different distances detected in PONI files:\n"
                    + "\n".join([f"  {d*100:.2f} cm" for d in distances_m[:5]])
                    + ("\n  ..." if len(distances_m) > 5 else "")
                    + "\n\nPlease verify and enter the correct distance.",
                )
        
        # Always prompt user to confirm/override distance
        if poni_distance_cm is not None:
            # Show dialog with PONI distance for user confirmation
            res = QMessageBox.question(
                self,
                "Confirm Distance from PONI",
                f"Distance from PONI file: {poni_distance_cm:.2f} cm\n\n"
                f"Use this distance, or enter a different value?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.Yes,
            )
            if res == QMessageBox.Yes:
                user_distance_cm = poni_distance_cm
            else:
                user_distance_cm = self._prompt_distance_cm(default_cm=poni_distance_cm)
        else:
            # No PONI distance available, must enter manually
            user_distance_cm = self._prompt_distance_cm()

        if user_distance_cm is None:
            return

        # Get technical temp folder for HDF5 generation
        tech_temp_folder = _get_technical_temp_folder(self.config if hasattr(self, "config") else None)
        
        # Generate HDF5 container in temp folder
        try:
            container_id, temp_file_path = technical_container.generate_from_aux_table(
                folder=tech_temp_folder,
                aux_measurements=aux_measurements,
                pony_data=pony_data,
                detector_config=self.config.get("detectors", []),
                active_detector_ids=self._get_active_detector_ids(),
                distance_cm=user_distance_cm,
                poni_distance_cm=poni_distance_cm,  # Real distance from PONI file
            )
        except Exception as e:
            QMessageBox.critical(
                self, "HDF5 Write Error", f"Failed to generate HDF5 container:\n{e}"
            )
            return

        self._log_technical_event(
            f"Technical HDF5 generated in temp: {os.path.basename(temp_file_path)}"
        )
        
        # Copy to user-specified storage folder
        import shutil
        try:
            storage_file_path = os.path.join(folder, os.path.basename(temp_file_path))
            shutil.copy2(temp_file_path, storage_file_path)
            self._log_technical_event(
                f"Copied to storage: {os.path.basename(storage_file_path)}"
            )
            final_path = storage_file_path
        except Exception as e:
            logger.warning(f"Failed to copy to storage folder: {e}")
            self._log_technical_event(
                f"Warning: Could not copy to storage folder, file remains in temp: {temp_file_path}"
            )
            final_path = temp_file_path

        QMessageBox.information(
            self,
            "HDF5 Generated",
            f"Temp location:\n{temp_file_path}\n\nStorage location:\n{final_path}\n\nContainer ID:\n{container_id}",
        )
