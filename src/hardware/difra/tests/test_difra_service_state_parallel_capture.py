import os
import sys
import time
import uuid
from pathlib import Path

import numpy as np

SRC_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if SRC_ROOT not in sys.path:
    sys.path.insert(0, SRC_ROOT)

from hardware.difra.grpc_server.difra_service_state import DifraServiceState


class _SleepyDetector:
    def __init__(self, alias: str):
        self.alias = alias
        self.size = (16, 16)

    def capture_point(self, Nframes, Nseconds, filename_base):
        time.sleep(float(Nseconds) * max(int(Nframes), 1))
        data = np.ones((8, 8), dtype=np.float64)
        np.savetxt(str(filename_base) + ".txt", data)
        return True


def _dummy_config(base_folder: Path):
    return {
        "DEV": True,
        "measurements_folder": str(base_folder),
        "detectors": [],
        "dev_active_detectors": [],
        "active_detectors": [],
        "translation_stages": [],
        "dev_active_stages": [],
        "active_translation_stages": [],
    }


def test_capture_detectors_runs_in_parallel(tmp_path: Path):
    state = DifraServiceState(config=_dummy_config(tmp_path))
    state.detector_controllers = {
        "A": _SleepyDetector("A"),
        "B": _SleepyDetector("B"),
    }

    run_id = str(uuid.uuid4())
    t0 = time.perf_counter()
    outputs = state._capture_detectors(run_id=run_id, exposure_time_ms=800)
    elapsed = time.perf_counter() - t0

    assert set(outputs.keys()) == {"A", "B"}
    assert Path(outputs["A"]).exists()
    assert Path(outputs["B"]).exists()
    # Sequential would be ~1.6s; parallel should stay near single-detector time.
    assert elapsed < 1.35
