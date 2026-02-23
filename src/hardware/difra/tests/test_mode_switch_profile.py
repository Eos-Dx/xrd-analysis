import os
import sys


SRC_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if SRC_ROOT not in sys.path:
    sys.path.insert(0, SRC_ROOT)

from hardware.difra.gui.main_window_ext.zone_measurements.detector_param_mixin import (
    DetectorParamMixin,
)


class _ModeSwitchHarness(DetectorParamMixin):
    def __init__(self):
        self.config = {"DEV": True}
        self.hardware_initialized = True
        self.hardware_client = object()
        self.hardware_controller = object()
        self.stage_controller = object()
        self.detector_controller = object()
        self.toggle_calls = 0
        self.refresh_tabs_calls = 0
        self.refresh_sidecar_calls = 0

    def toggle_hardware(self):
        self.toggle_calls += 1
        self.hardware_initialized = False

    def refresh_detector_tabs_for_mode_switch(self):
        self.refresh_tabs_calls += 1

    def refresh_sidecar_status(self, show_message: bool = False):
        self.refresh_sidecar_calls += 1
        return True


def test_mode_switch_forces_runtime_reset_before_next_init():
    ui = _ModeSwitchHarness()
    ui.on_config_mode_changed(False)

    assert ui.config["DEV"] is False
    assert ui.toggle_calls == 1
    assert ui.refresh_tabs_calls == 1
    assert ui.refresh_sidecar_calls == 1
    assert ui.hardware_initialized is False
    assert ui.hardware_client is None
    assert ui.hardware_controller is None
    assert ui.stage_controller is None
    assert ui.detector_controller is None
