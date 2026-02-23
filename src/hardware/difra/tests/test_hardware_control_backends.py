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


def test_production_mode_uses_active_profiles_not_dev_profiles():
    cfg = {
        "DEV": False,
        "detectors": [
            {
                "alias": "PROD_DET",
                "id": "PROD-DET-001",
                "type": "DummyDetector",
                "size": {"width": 16, "height": 16},
            },
            {
                "alias": "DEV_DET",
                "id": "DEV-DET-001",
                "type": "DummyDetector",
                "size": {"width": 16, "height": 16},
            },
        ],
        "active_detectors": ["PROD-DET-001"],
        "dev_active_detectors": ["DEV-DET-001"],
        "translation_stages": [
            {"alias": "PROD_STAGE", "id": "PROD-STAGE-001", "type": "DummyStage"},
            {"alias": "DEV_STAGE", "id": "DEV-STAGE-001", "type": "DummyStage"},
        ],
        "active_translation_stages": ["PROD-STAGE-001"],
        "dev_active_stages": ["DEV-STAGE-001"],
    }

    controller = HardwareController(cfg)
    stage_ok, detector_ok = controller.initialize(init_stage=True, init_detector=True)

    assert stage_ok is True
    assert detector_ok is True
    assert controller.stage_controller is not None
    assert getattr(controller.stage_controller, "alias", "") == "PROD_STAGE"
    assert "PROD_DET" in controller.detectors
    assert "DEV_DET" not in controller.detectors
