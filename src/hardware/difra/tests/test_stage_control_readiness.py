import os
import sys
import importlib.util
from pathlib import Path

SRC_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if SRC_ROOT not in sys.path:
    sys.path.insert(0, SRC_ROOT)

_MIXIN_PATH = (
    Path(__file__).resolve().parents[1]
    / "gui"
    / "main_window_ext"
    / "zone_measurements"
    / "logic"
    / "stage_control_mixin.py"
)
_SPEC = importlib.util.spec_from_file_location(
    "stage_control_mixin_for_tests",
    _MIXIN_PATH,
)
_MODULE = importlib.util.module_from_spec(_SPEC)
assert _SPEC and _SPEC.loader
_SPEC.loader.exec_module(_MODULE)
StageControlMixin = _MODULE.StageControlMixin

from hardware.difra.hardware.hardware_client import CommandReadiness


class _StubWidget:
    def __init__(self):
        self.enabled = False
        self.style = ""

    def setEnabled(self, enabled):
        self.enabled = bool(enabled)

    def setStyleSheet(self, style):
        self.style = style


class _StubClient:
    def __init__(self, readiness):
        self._readiness = readiness

    def get_command_readiness(self):
        return self._readiness


class _Harness(StageControlMixin):
    def __init__(self, readiness):
        self.hardware_client = _StubClient(readiness)
        self.start_btn = _StubWidget()
        self.pause_btn = _StubWidget()
        self.stop_btn = _StubWidget()
        self.xPosSpin = _StubWidget()
        self.yPosSpin = _StubWidget()
        self.gotoBtn = _StubWidget()
        self.homeBtn = _StubWidget()
        self.loadPosBtn = _StubWidget()


def test_stage_control_applies_command_readiness_to_ui_controls():
    readiness = {
        ("Motion", "MoveTo"): CommandReadiness(
            ready=False, reasons=["Motion stage is not initialized"]
        ),
        ("Motion", "Home"): CommandReadiness(ready=True, reasons=[]),
        ("Acquisition", "StartExposure"): CommandReadiness(
            ready=False, reasons=["Detector is not initialized"]
        ),
    }
    harness = _Harness(readiness)

    harness._apply_readiness_to_controls(hardware_ok=True)

    assert harness.start_btn.enabled is False
    assert harness.gotoBtn.enabled is False
    assert harness.xPosSpin.enabled is False
    assert harness.yPosSpin.enabled is False
    assert harness.homeBtn.enabled is True
