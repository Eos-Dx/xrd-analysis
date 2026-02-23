import os
import socket
import subprocess
import sys
import time
from pathlib import Path


SRC_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if SRC_ROOT not in sys.path:
    sys.path.insert(0, SRC_ROOT)

from hardware.difra.gui.main_window_ext.zone_measurements.logic.stage_control_mixin import (
    StageControlMixin,
)


REPO_ROOT = Path(__file__).resolve().parents[4]
SIDECAR_SCRIPT = REPO_ROOT / "src" / "hardware" / "difra" / "scripts" / "pixet_sidecar_server.py"


def _free_tcp_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def test_sidecar_required_only_for_pixet_detectors():
    pixet_cfg = {
        "DEV": False,
        "detectors": [{"id": "P1", "type": "Pixet"}],
        "active_detectors": ["P1"],
        "dev_active_detectors": [],
    }
    dummy_cfg = {
        "DEV": True,
        "detectors": [{"id": "D1", "type": "DummyDetector"}],
        "active_detectors": [],
        "dev_active_detectors": ["D1"],
    }

    assert StageControlMixin._is_sidecar_required_for_config(pixet_cfg) is True
    assert StageControlMixin._is_sidecar_required_for_config(dummy_cfg) is False


def test_sidecar_ping_probe_reports_alive():
    port = _free_tcp_port()
    proc = subprocess.Popen(
        [
            sys.executable,
            "-u",
            str(SIDECAR_SCRIPT),
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    try:
        for _ in range(60):
            alive, _reason, _latency_ms = StageControlMixin._probe_sidecar_endpoint(
                "127.0.0.1", port, timeout_s=0.2
            )
            if alive:
                break
            time.sleep(0.05)
        else:
            assert False, "sidecar did not become reachable for ping probe"
    finally:
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=2.0)
