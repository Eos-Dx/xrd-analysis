# zone_measurements/logic/process_mixin.py

from PyQt5.QtCore import QThread, QTimer
from PyQt5.QtGui import QColor
from PyQt5.QtWidgets import QMessageBox

from hardware.difra.gui.main_window_ext.zone_measurements.logic.process_capture_mixin import (
    ZoneMeasurementsProcessCaptureMixin,
)
from hardware.difra.gui.main_window_ext.zone_measurements.logic.process_results_mixin import (
    ZoneMeasurementsProcessResultsMixin,
)
from hardware.difra.gui.main_window_ext.zone_measurements.logic.process_start_mixin import (
    ZoneMeasurementsProcessStartMixin,
)
from hardware.difra.utils.logger import get_module_logger

# Defer all technical imports to avoid pyFAI crashes on startup
_ZONE_TECHNICAL_IMPORTS_AVAILABLE = None
_zone_technical_modules = {}


def _get_zone_technical_imports():
    """Lazy import of technical modules to avoid startup crashes."""
    global _ZONE_TECHNICAL_IMPORTS_AVAILABLE, _zone_technical_modules

    if _ZONE_TECHNICAL_IMPORTS_AVAILABLE is not None:
        return _ZONE_TECHNICAL_IMPORTS_AVAILABLE

    try:
        from hardware.difra.gui.technical.capture import CaptureWorker, validate_folder
        from hardware.difra.gui.technical.measurement_worker import MeasurementWorker
        from hardware.difra.gui.technical.widgets import MeasurementHistoryWidget

        _zone_technical_modules.update(
            {
                "CaptureWorker": CaptureWorker,
                "validate_folder": validate_folder,
                "MeasurementWorker": MeasurementWorker,
                "MeasurementHistoryWidget": MeasurementHistoryWidget,
            }
        )
        _ZONE_TECHNICAL_IMPORTS_AVAILABLE = True
        return True
    except Exception as e:
        print(f"Warning: Zone technical measurement imports failed: {e}")
        print("Zone measurements will be disabled.")
        _ZONE_TECHNICAL_IMPORTS_AVAILABLE = False
        return False


def _get_zone_technical_module(name):
    """Get a zone technical module by name, with fallback stubs."""
    if _get_zone_technical_imports():
        return _zone_technical_modules.get(name)

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
        "MeasurementHistoryWidget": type(
            "MeasurementHistoryWidget",
            (),
            {"__init__": lambda self, *args, **kwargs: None},
        ),
    }
    return stubs.get(name)


logger = get_module_logger(__name__)


class ZoneMeasurementsProcessMixin(
    ZoneMeasurementsProcessStartMixin,
    ZoneMeasurementsProcessCaptureMixin,
    ZoneMeasurementsProcessResultsMixin,
):
    def _zone_technical_imports_available(self):
        return _get_zone_technical_imports()

    def _get_zone_technical_module(self, name):
        return _get_zone_technical_module(name)
