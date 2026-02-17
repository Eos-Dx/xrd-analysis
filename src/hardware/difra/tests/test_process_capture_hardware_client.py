import os
import sys
import importlib.util
from pathlib import Path
import types

SRC_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if SRC_ROOT not in sys.path:
    sys.path.insert(0, SRC_ROOT)

_MIXIN_PATH = (
    Path(__file__).resolve().parents[1]
    / "gui"
    / "main_window_ext"
    / "zone_measurements"
    / "logic"
    / "process_capture_mixin.py"
)
_SPEC = importlib.util.spec_from_file_location(
    "process_capture_mixin_for_tests",
    _MIXIN_PATH,
)
_MODULE = importlib.util.module_from_spec(_SPEC)

_container_api_stub = types.ModuleType("hardware.difra.gui.container_api")
_container_api_stub.get_container_version = lambda *_args, **_kwargs: "0.2"
_original_container_api_module = sys.modules.get("hardware.difra.gui.container_api")
sys.modules["hardware.difra.gui.container_api"] = _container_api_stub

assert _SPEC and _SPEC.loader
_SPEC.loader.exec_module(_MODULE)
if _original_container_api_module is not None:
    sys.modules["hardware.difra.gui.container_api"] = _original_container_api_module
else:
    del sys.modules["hardware.difra.gui.container_api"]
ZoneMeasurementsProcessCaptureMixin = _MODULE.ZoneMeasurementsProcessCaptureMixin


class _StubStageController:
    def __init__(self):
        self.calls = []

    def move_stage(self, x, y, move_timeout=20):
        self.calls.append((x, y, move_timeout))
        return x, y


class _StubHardwareClient:
    def __init__(self):
        self.calls = []
        self._x = 0.0
        self._y = 0.0

    def move_to(self, position_mm, axis, timeout_s=25.0):
        self.calls.append((position_mm, axis, timeout_s))
        if axis == "x":
            self._x = float(position_mm)
        elif axis == "y":
            self._y = float(position_mm)
        else:
            raise ValueError(f"Unexpected axis: {axis}")
        return self._x, self._y


class _Harness(ZoneMeasurementsProcessCaptureMixin):
    pass


def test_move_stage_prefers_hardware_client():
    h = _Harness()
    h.hardware_client = _StubHardwareClient()
    h.stage_controller = _StubStageController()

    out = h._move_stage(1.5, -2.0, timeout_s=7.0)
    assert out == (1.5, -2.0)
    assert h.hardware_client.calls == [
        (1.5, "x", 7.0),
        (-2.0, "y", 7.0),
    ]
    assert h.stage_controller.calls == []


def test_move_stage_falls_back_to_stage_controller():
    h = _Harness()
    h.hardware_client = None
    h.stage_controller = _StubStageController()

    out = h._move_stage(3.0, 4.0, timeout_s=9.0)
    assert out == (3.0, 4.0)
    assert h.stage_controller.calls == [(3.0, 4.0, 9.0)]
