# zone_measurements/zone_measurements_mixin.py
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtWidgets import QDockWidget, QTabWidget, QVBoxLayout, QWidget

from .attenuation_mixin import AttenuationMixin
from .detector_param_mixin import DetectorParamMixin
from .session_tab_mixin import SessionTabMixin
from .zone_measurements_logic_mixin import ZoneMeasurementsLogicMixin


class ZoneMeasurementsMixin(
    ZoneMeasurementsLogicMixin, DetectorParamMixin, AttenuationMixin, SessionTabMixin
):
    """
    Aggregator mixin that combines measurement, detector param, and attenuation logic.
    Inherit from this in your main window.
    """

    hardware_state_changed = pyqtSignal(bool)

    def create_zone_measurements(self):
        """
        Entrypoint for all measurement-related UI setup.
        Wraps the tab widget in a Dock. Call this ONCE in your MainWindow __init__!
        """
        # --- Container for tabs ---
        container = QWidget()
        layout = QVBoxLayout(container)
        self.tabs = QTabWidget()
        layout.addWidget(self.tabs)

        # --- Add tabs ---
        self.create_zone_measurements_widget()  # Adds the Measurements tab (and its controls) to self.tabs
        self.create_attenuation_tab()  # Adds the Attenuation tab to self.tabs
        self.setup_detector_param_tabs()  # Initialize detector tabs tracking
        self.populate_detector_param_tabs()  # Creates one tab per active detector alias
        self.create_session_tab()  # Adds the Session tab to self.tabs

        # --- Create and set up the Dock ---
        self.zoneMeasurementsDock = QDockWidget("Zone Measurements", self)
        self.zoneMeasurementsDock.setObjectName("ZoneMeasurementsDock")
        self.zoneMeasurementsDock.setWidget(container)
        
        # Set minimum height to be compact - just enough for controls and a few rows
        # This gives more vertical space to other zones (image view, etc.)
        try:
            # Minimum: title bar (~20px) + tab bar (~30px) + controls (~150px) + 2-3 table rows (~100px)
            self.zoneMeasurementsDock.setMinimumHeight(150)
        except Exception:
            pass
        
        self.addDockWidget(Qt.BottomDockWidgetArea, self.zoneMeasurementsDock)
