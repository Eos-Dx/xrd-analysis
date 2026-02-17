import os
import sys
import threading
import time
from pathlib import Path

SRC_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if SRC_ROOT not in sys.path:
    sys.path.insert(0, SRC_ROOT)

from hardware.difra.scripts import pixet_sidecar_server as sidecar


def _init_dummy(alias: str):
    return sidecar._dispatch(
        "init_detector",
        {
            "alias": alias,
            "detector_type": "DummyDetector",
            "size": [16, 16],
            "config": {"type": "DummyDetector"},
        },
    )


def _capture(alias: str, base: str, seconds: float):
    return sidecar._dispatch(
        "capture_point",
        {
            "alias": alias,
            "Nframes": 1,
            "Nseconds": seconds,
            "filename_base": base,
            "detector_type": "DummyDetector",
            "size": [16, 16],
            "config": {"type": "DummyDetector"},
        },
    )


def test_sidecar_capture_parallel_for_different_aliases(tmp_path: Path):
    sidecar.STATE.controllers.clear()
    sidecar.STATE.controller_locks.clear()

    _init_dummy("D1")
    _init_dummy("D2")

    barrier = threading.Barrier(3)
    errors = []

    def worker(alias: str):
        try:
            barrier.wait(timeout=5.0)
            _capture(alias, str(tmp_path / alias), 0.8)
        except Exception as exc:  # pragma: no cover - diagnostics in case of failure
            errors.append(exc)

    t1 = threading.Thread(target=worker, args=("D1",))
    t2 = threading.Thread(target=worker, args=("D2",))
    t1.start()
    t2.start()

    barrier.wait(timeout=5.0)
    t0 = time.perf_counter()
    t1.join(timeout=5.0)
    t2.join(timeout=5.0)
    elapsed = time.perf_counter() - t0

    assert not errors
    assert (tmp_path / "D1.txt").exists()
    assert (tmp_path / "D2.txt").exists()
    # Sequential would be ~1.6s due 0.8s + 0.8s.
    assert elapsed < 1.35

    sidecar._dispatch("shutdown", {})


def test_sidecar_legacy_capture_parallel_for_different_aliases(tmp_path: Path):
    class _FakeLegacyDetector:
        def __init__(self, alias, size=(16, 16), config=None):
            self.alias = alias
            self.size = size
            self.config = config or {}

        def init_detector(self):
            return True

        def deinit_detector(self):
            return True

        def capture_point(self, Nframes, Nseconds, filename_base):
            time.sleep(float(Nseconds))
            Path(f"{filename_base}.txt").write_text("ok", encoding="utf-8")
            return True

    original_legacy = sidecar.PixetLegacyDetectorController
    sidecar.PixetLegacyDetectorController = _FakeLegacyDetector
    try:
        sidecar.STATE.controllers.clear()
        sidecar.STATE.controller_locks.clear()

        sidecar._dispatch(
            "init_detector",
            {
                "alias": "L1",
                "detector_type": "Pixet",
                "size": [16, 16],
                "config": {"type": "Pixet"},
            },
        )
        sidecar._dispatch(
            "init_detector",
            {
                "alias": "L2",
                "detector_type": "Pixet",
                "size": [16, 16],
                "config": {"type": "Pixet"},
            },
        )

        barrier = threading.Barrier(3)
        errors = []

        def worker(alias: str):
            try:
                barrier.wait(timeout=5.0)
                sidecar._dispatch(
                    "capture_point",
                    {
                        "alias": alias,
                        "Nframes": 1,
                        "Nseconds": 0.8,
                        "filename_base": str(tmp_path / alias),
                        # Non-dummy type forces legacy branch in sidecar.
                        "detector_type": "Pixet",
                        "size": [16, 16],
                        "config": {"type": "Pixet"},
                    },
                )
            except Exception as exc:  # pragma: no cover - diagnostics in case of failure
                errors.append(exc)

        t1 = threading.Thread(target=worker, args=("L1",))
        t2 = threading.Thread(target=worker, args=("L2",))
        t1.start()
        t2.start()

        barrier.wait(timeout=5.0)
        t0 = time.perf_counter()
        t1.join(timeout=5.0)
        t2.join(timeout=5.0)
        elapsed = time.perf_counter() - t0

        assert not errors
        assert (tmp_path / "L1.txt").exists()
        assert (tmp_path / "L2.txt").exists()
        assert elapsed < 1.35
    finally:
        sidecar._dispatch("shutdown", {})
        sidecar.PixetLegacyDetectorController = original_legacy
