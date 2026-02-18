import logging
import traceback

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

    QEvent = object

    class _Qt:
        LeftDockWidgetArea = 1
        RightDockWidgetArea = 2
        Horizontal = 0
        Key_Delete = 16777223
        UserRole = 32
        AlignCenter = 0
        ItemIsSelectable = 1
        ItemIsEnabled = 2

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

        def stop(self):
            pass

    QTimer = _QTimer

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
        Yes, No, Cancel = 1, 0, -1

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


try:
    from hardware.difra.gui.main_window_ext.zone_measurements import (
        ZoneMeasurementsMixin as _ZoneMeasurementsMixin,
    )
except Exception:  # pragma: no cover - test stubs
    class _ZoneMeasurementsMixin(object):
        pass


_TECHNICAL_IMPORTS_AVAILABLE = None
_technical_modules = {}


def _get_technical_imports():
    """Lazy import of technical modules to avoid startup crashes."""
    global _TECHNICAL_IMPORTS_AVAILABLE, _technical_modules
    if _TECHNICAL_IMPORTS_AVAILABLE is not None:
        return _TECHNICAL_IMPORTS_AVAILABLE

    try:
        from hardware.difra.gui.technical.capture import CaptureWorker, show_measurement_window, validate_folder
        from hardware.difra.gui.technical.measurement_worker import MeasurementWorker

        _technical_modules.update(
            {
                "CaptureWorker": CaptureWorker,
                "show_measurement_window": show_measurement_window,
                "validate_folder": validate_folder,
                "MeasurementWorker": MeasurementWorker,
            }
        )
        _TECHNICAL_IMPORTS_AVAILABLE = True
        logger.info("Technical measurement imports successful")
        return True
    except Exception as e:
        tb_str = "".join(traceback.format_exception(type(e), e, e.__traceback__))
        logger.error(
            f"Technical measurement imports failed: {type(e).__name__}: {e}\n{tb_str}",
            exc_info=True,
        )
        _TECHNICAL_IMPORTS_AVAILABLE = False
        return False


def _get_technical_module(name):
    """Get a technical module by name, with fallback stubs."""
    if _get_technical_imports():
        return _technical_modules.get(name)

    stubs = {
        "CaptureWorker": type(
            "CaptureWorker",
            (),
            {
                "__init__": lambda self, *args, **kwargs: None,
                "moveToThread": lambda self, thread: None,
                "finished": type("Signal", (), {"connect": lambda self, f: None})(),
            },
        ),
        "show_measurement_window": lambda *args, **kwargs: print(
            "Technical measurement window not available - imports failed"
        ),
        "validate_folder": lambda path: str(path) if path else "",
        "MeasurementWorker": type(
            "MeasurementWorker",
            (),
            {
                "__init__": lambda self, *args, **kwargs: None,
                "run": lambda self: None,
                "add_aux_item": type("Signal", (), {"connect": lambda self, f: None})(),
            },
        ),
    }
    return stubs.get(name)


from .technical.aux_table_mixin import TechnicalAuxTableMixin
from .technical.capture_mixin import TechnicalCaptureMixin
from .technical.h5_generation_mixin import H5GenerationMixin
from .technical.h5_management_mixin import H5ManagementMixin
from .technical.helpers import (
    _get_default_folder,
    _get_difra_base_folder,
    _get_measurement_default_folder,
    _get_technical_archive_folder,
    _get_technical_storage_folder,
    _get_technical_temp_folder,
)
from .technical.panel_mixin import TechnicalPanelMixin
from .technical.realtime_mixin import TechnicalRealtimeMixin


class PoniFileSelectionDialog(QDialog):
    """Dialog for selecting PONI files for each detector alias."""

    def __init__(self, aliases, current_poni_files=None, parent=None):
        super().__init__(parent)
        self.aliases = aliases
        self.poni_files = {}
        self.line_edits = {}

        if current_poni_files:
            for alias in aliases:
                if alias in current_poni_files:
                    poni_info = current_poni_files[alias]
                    if isinstance(poni_info, dict) and "path" in poni_info:
                        self.poni_files[alias] = poni_info["path"]
                    elif hasattr(self.parent(), "poni_files") and alias in self.parent().poni_files:
                        parent_poni = self.parent().poni_files[alias]
                        if isinstance(parent_poni, dict) and "path" in parent_poni:
                            self.poni_files[alias] = parent_poni["path"]

        self.setup_ui()

    def setup_ui(self):
        self.setWindowTitle("Select PONI Files for Technical Meta")
        self.setModal(True)
        self.resize(600, 400)

        layout = QVBoxLayout(self)
        header = QLabel("Select PONI calibration files for each detector alias:")
        header.setStyleSheet("font-weight: bold; margin-bottom: 10px;")
        layout.addWidget(header)
        form_layout = QFormLayout()

        for alias in self.aliases:
            h_layout = QHBoxLayout()
            line_edit = QLineEdit()
            line_edit.setPlaceholderText(f"Select PONI file for {alias}")
            if alias in self.poni_files:
                line_edit.setText(self.poni_files[alias])
            self.line_edits[alias] = line_edit
            h_layout.addWidget(line_edit)

            browse_btn = QPushButton("Browse...")
            browse_btn.clicked.connect(lambda checked, a=alias: self.browse_poni_file(a))
            h_layout.addWidget(browse_btn)

            clear_btn = QPushButton("Clear")
            clear_btn.clicked.connect(lambda checked, a=alias: self.clear_poni_file(a))
            h_layout.addWidget(clear_btn)
            form_layout.addRow(f"{alias}:", h_layout)

        layout.addLayout(form_layout)
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
        self.line_edits[alias].setText("")
        if alias in self.poni_files:
            del self.poni_files[alias]

    def get_poni_files(self):
        result = {}
        for alias, line_edit in self.line_edits.items():
            path = line_edit.text().strip()
            if path:
                result[alias] = path
        return result


class TechnicalMeasurementsMixin(
    TechnicalPanelMixin,
    TechnicalCaptureMixin,
    TechnicalAuxTableMixin,
    TechnicalRealtimeMixin,
    H5GenerationMixin,
    H5ManagementMixin,
    _ZoneMeasurementsMixin,
):
    AUX_COL_PRIMARY = 0
    AUX_COL_FILE = 1
    AUX_COL_TYPE = 2
    AUX_COL_ALIAS = 3

    NO_SELECTION_LABEL = "— Select —"
    TYPE_OPTIONS = ["AGBH", "DARK", "EMPTY", "BACKGROUND", "SPECIAL"]
    REQUIRED_TYPE_OPTIONS = ["AGBH", "DARK", "EMPTY", "BACKGROUND"]

    def _technical_imports_available(self):
        return _get_technical_imports()

    def _get_technical_module(self, name):
        return _get_technical_module(name)

    def _log_technical_event(self, message: str):
        try:
            self._append_measurement_log(f"[TECH] {message}")
        except Exception:
            print(f"[TECH] {message}")
