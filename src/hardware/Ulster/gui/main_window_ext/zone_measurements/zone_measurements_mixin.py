# zone_measurements/zone_measurements_mixin.py
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtWidgets import QDockWidget, QTabWidget, QVBoxLayout, QWidget

from .attenuation_mixin import AttenuationMixin
from .detector_param_mixin import DetectorParamMixin
from .zone_measurements_logic_mixin import ZoneMeasurementsLogicMixin


class ZoneMeasurementsMixin(
    ZoneMeasurementsLogicMixin, DetectorParamMixin, AttenuationMixin
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
        self.setup_detector_param_tab()  # Adds Detector param tab (if not present)
        self.populate_detector_param_tab()  # Populates detector param tab

        # --- Create and set up the Dock ---
        self.zoneMeasurementsDock = QDockWidget("Zone Measurements", self)
        self.zoneMeasurementsDock.setWidget(container)
        self.addDockWidget(Qt.BottomDockWidgetArea, self.zoneMeasurementsDock)
