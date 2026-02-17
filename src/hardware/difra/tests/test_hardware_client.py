import os
import sys
import time
from pathlib import Path

import pytest

SRC_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if SRC_ROOT not in sys.path:
    sys.path.insert(0, SRC_ROOT)

from hardware.difra.hardware.hardware_client import (
    DirectHardwareClient,
    DualPathHardwareClient,
    create_hardware_client,
)


def _dummy_config():
    return {
        "DEV": True,
        "detectors": [
            {
                "alias": "PRIMARY",
                "type": "DummyDetector",
                "id": "DUMMY-DET-1",
                "size": {"width": 32, "height": 32},
            }
        ],
        "dev_active_detectors": ["DUMMY-DET-1"],
        "active_detectors": [],
        "translation_stages": [
            {
                "alias": "DUMMY_STAGE",
                "type": "DummyStage",
                "id": "DUMMY-STAGE-1",
                "settings": {
                    "limits_mm": {
                        "x": [-14.0, 14.0],
                        "y": [-14.0, 14.0],
                    },
                    "home": [1.0, 2.0],
                    "load": [-3.0, -4.0],
                },
            }
        ],
        "dev_active_stages": ["DUMMY-STAGE-1"],
        "active_translation_stages": [],
    }


def test_direct_hardware_client_motion_and_detector_flow():
    client = DirectHardwareClient(_dummy_config())

    assert client.initialize_motion() is True
    assert client.initialize_detector() is True

    x, y = client.move_to(2.5, axis="x", timeout_s=2.0)
    assert x == pytest.approx(2.5, abs=1e-6)
    x, y = client.move_to(-1.5, axis="y", timeout_s=2.0)
    assert x == pytest.approx(2.5, abs=1e-6)
    assert y == pytest.approx(-1.5, abs=1e-6)

    cur_x, cur_y = client.get_xy_position()
    assert cur_x == pytest.approx(2.5, abs=1e-6)
    assert cur_y == pytest.approx(-1.5, abs=1e-6)

    readiness = client.get_command_readiness()
    assert readiness[("Motion", "MoveTo")].ready is True
    assert readiness[("Acquisition", "StartExposure")].ready is True

    client.deinitialize()
    readiness_after = client.get_command_readiness()
    assert readiness_after[("Motion", "MoveTo")].ready is False

    state = client.get_state()
    assert state["locks"]["device_locked"] is False
    assert state["locks"]["session_locked"] is False
    assert state["locks"]["technical_container_locked"] is False


def test_dual_path_client_falls_back_to_direct_when_grpc_unavailable():
    config = _dummy_config()
    config["hardware_protocol"] = {
        "client_mode": "dual",
        "grpc_host": "127.0.0.1",
        "grpc_port": 65531,
        "grpc_timeout_s": 0.2,
    }

    client = create_hardware_client(config)
    assert isinstance(client, DualPathHardwareClient)

    assert client.initialize_motion() is True
    assert client.initialize_detector() is True
    assert client.last_backend == "direct"

    x, y = client.move_to(1.0, axis="x", timeout_s=1.0)
    assert x == pytest.approx(1.0, abs=1e-6)
    x, y = client.move_to(1.0, axis="y", timeout_s=1.0)
    assert x == pytest.approx(1.0, abs=1e-6)
    assert y == pytest.approx(1.0, abs=1e-6)


def test_direct_capture_exposure_runs_detectors_in_parallel():
    class _SleepDetector:
        def __init__(self, sleep_s: float):
            self.sleep_s = float(sleep_s)

        def capture_point(self, Nframes, Nseconds, filename_base):
            time.sleep(self.sleep_s)
            Path(f"{filename_base}.txt").write_text("ok", encoding="utf-8")
            return True

    client = DirectHardwareClient(_dummy_config())
    client._controller.detectors = {
        "PRIMARY": _SleepDetector(0.8),
        "SECONDARY": _SleepDetector(0.8),
    }

    started = time.perf_counter()
    outputs = client.capture_exposure(exposure_s=0.8, frames=1, timeout_s=5.0)
    elapsed = time.perf_counter() - started

    assert set(outputs.keys()) == {"PRIMARY", "SECONDARY"}
    # Sequential would be close to ~1.6s; parallel should stay near single-detector time.
    assert elapsed < 1.35
