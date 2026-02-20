import os
import socket
import subprocess
import sys
import time
from pathlib import Path


SRC_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if SRC_ROOT not in sys.path:
    sys.path.insert(0, SRC_ROOT)

REPO_ROOT = Path(__file__).resolve().parents[4]
SIDECAR_SCRIPT = REPO_ROOT / "src" / "hardware" / "difra" / "scripts" / "pixet_sidecar_server.py"


def _free_tcp_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _wait_for_port(host: str, port: int, timeout_s: float = 5.0) -> None:
    deadline = time.time() + timeout_s
    last_error = None
    while time.time() < deadline:
        try:
            with socket.create_connection((host, port), timeout=0.2):
                return
        except OSError as exc:
            last_error = exc
            time.sleep(0.05)
    raise AssertionError(f"Timed out waiting for {host}:{port} to accept connections: {last_error}")


def test_sidecar_exits_when_owner_process_exits():
    owner_proc = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(0.8)"]
    )
    port = _free_tcp_port()
    sidecar_proc = subprocess.Popen(
        [
            sys.executable,
            "-u",
            str(SIDECAR_SCRIPT),
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "--owner-pid",
            str(owner_proc.pid),
            "--owner-check-interval-s",
            "0.1",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )

    try:
        _wait_for_port("127.0.0.1", port, timeout_s=5.0)
        owner_proc.wait(timeout=5.0)
        output, _ = sidecar_proc.communicate(timeout=8.0)
        assert sidecar_proc.returncode == 0
        assert "owner pid" in output
    finally:
        if owner_proc.poll() is None:
            owner_proc.terminate()
            try:
                owner_proc.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                owner_proc.kill()
                owner_proc.wait(timeout=2.0)

        if sidecar_proc.poll() is None:
            sidecar_proc.terminate()
            try:
                sidecar_proc.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                sidecar_proc.kill()
                sidecar_proc.wait(timeout=2.0)


def test_sidecar_with_live_owner_does_not_exit_immediately():
    owner_proc = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(2.0)"]
    )
    port = _free_tcp_port()
    sidecar_proc = subprocess.Popen(
        [
            sys.executable,
            "-u",
            str(SIDECAR_SCRIPT),
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "--owner-pid",
            str(owner_proc.pid),
            "--owner-check-interval-s",
            "0.1",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )

    try:
        _wait_for_port("127.0.0.1", port, timeout_s=5.0)
        time.sleep(0.6)
        assert sidecar_proc.poll() is None
    finally:
        if owner_proc.poll() is None:
            owner_proc.terminate()
            try:
                owner_proc.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                owner_proc.kill()
                owner_proc.wait(timeout=2.0)

        if sidecar_proc.poll() is None:
            sidecar_proc.terminate()
            try:
                sidecar_proc.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                sidecar_proc.kill()
                sidecar_proc.wait(timeout=2.0)
