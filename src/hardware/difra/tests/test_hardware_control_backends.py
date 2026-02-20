import os
import sys


SRC_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if SRC_ROOT not in sys.path:
    sys.path.insert(0, SRC_ROOT)

from hardware.difra.hardware.detectors import DummyDetectorController
from hardware.difra.hardware.hardware_control import HardwareController


def test_dummy_detector_stays_local_in_sidecar_backend(monkeypatch):
    monkeypatch.setenv("DETECTOR_BACKEND", "sidecar")
    monkeypatch.setenv("PIXET_BACKEND", "sidecar")

    cfg = {
        "DEV": True,
        "detectors": [
            {
                "alias": "DUMMY_A",
                "id": "DUMMY-001",
                "type": "DummyDetector",
                "size": {"width": 16, "height": 16},
            }
        ],
        "dev_active_detectors": ["DUMMY-001"],
        "active_detectors": [],
        "translation_stages": [],
        "dev_active_stages": [],
        "active_translation_stages": [],
    }

    controller = HardwareController(cfg)
    stage_ok, detector_ok = controller.initialize(init_stage=False, init_detector=True)

    assert stage_ok is False
    assert detector_ok is True
    assert "DUMMY_A" in controller.detectors
    assert isinstance(controller.detectors["DUMMY_A"], DummyDetectorController)
