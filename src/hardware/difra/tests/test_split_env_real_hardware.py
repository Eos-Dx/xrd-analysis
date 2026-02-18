"""
Test real hardware with split environments:
- ulster37 for legacy Pixet detectors (via sidecar)
- eosdx13 for modern stage control

This test validates the dual-environment hardware stack works properly.
"""
from __future__ import annotations

import contextlib
import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, Iterator

import pytest

SRC_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if SRC_ROOT not in sys.path:
    sys.path.insert(0, SRC_ROOT)

from hardware.difra.hardware.hardware_client import DirectHardwareClient

REPO_ROOT = Path(__file__).resolve().parents[4]
SIDECAR_SCRIPT = REPO_ROOT / "src" / "hardware" / "difra" / "scripts" / "pixet_sidecar_server.py"
SETUP_CONFIG = REPO_ROOT / "src" / "hardware" / "difra" / "resources" / "config" / "setups" / "Ulster (Xena).json"


def _read_json(path: Path) -> Dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _free_tcp_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _sidecar_ping(host: str, port: int, timeout_s: float = 0.5) -> bool:
    """Ping the sidecar to check if it's responsive."""
    payload = {"id": "ping", "cmd": "ping", "args": {}}
    raw = (json.dumps(payload) + "\n").encode("utf-8")
    try:
        with socket.create_connection((host, port), timeout=timeout_s) as sock:
            sock.settimeout(timeout_s)
            sock.sendall(raw)
            line = b""
            while b"\n" not in line:
                chunk = sock.recv(4096)
                if not chunk:
                    return False
                line += chunk
        response = json.loads(line.split(b"\n", 1)[0].decode("utf-8"))
        return bool(response.get("ok"))
    except (OSError, json.JSONDecodeError):
        return False


@contextlib.contextmanager
def _temporary_env(overrides: Dict[str, str]) -> Iterator[None]:
    """Temporarily set environment variables."""
    original = {k: os.environ.get(k) for k in overrides}
    try:
        for key, value in overrides.items():
            os.environ[key] = str(value)
        yield
    finally:
        for key, old in original.items():
            if old is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = old


@contextlib.contextmanager
def _started_sidecar_ulster37(host: str, port: int) -> Iterator[subprocess.Popen]:
    """
    Start the Pixet sidecar server using ulster37 conda environment.
    This environment has the legacy Pixet SDK Python bindings.
    """
    conda_exe = "conda"
    
    env = dict(os.environ)
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = f"{SRC_ROOT}{os.pathsep}{existing}" if existing else SRC_ROOT
    
    sidecar_cmd = [
        conda_exe,
        "run",
        "--live-stream",
        "--no-capture-output",
        "-n",
        "ulster37",
        "python",
        "-u",
        str(SIDECAR_SCRIPT),
        "--host",
        host,
        "--port",
        str(port),
    ]
    
    print(f"[INFO] Starting sidecar with ulster37: {' '.join(sidecar_cmd)}")
    
    proc = subprocess.Popen(
        sidecar_cmd,
        cwd=str(REPO_ROOT),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    
    try:
        # Wait for sidecar to become ready
        deadline = time.time() + 45.0
        last_output = []
        while time.time() < deadline:
            if proc.poll() is not None:
                out = proc.stdout.read() if proc.stdout else ""
                raise RuntimeError(
                    f"Sidecar exited early with code {proc.returncode}\n"
                    f"Last output:\n{out}"
                )
            try:
                if _sidecar_ping(host, port, timeout_s=1.0):
                    print(f"[INFO] Sidecar ready on {host}:{port}")
                    break
            except OSError:
                pass
            
            # Read some output for debugging (non-blocking not available on Windows pipes)
            # Just collect output for error reporting
            
            time.sleep(0.2)
        else:
            out_lines = "".join(last_output[-20:]) if last_output else "<no output>"
            raise TimeoutError(
                f"Timed out waiting for sidecar readiness after 45s\n"
                f"Recent output:\n{out_lines}"
            )
        
        yield proc
        
    finally:
        if proc.poll() is None:
            print("[INFO] Terminating sidecar...")
            proc.terminate()
            try:
                proc.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                print("[WARN] Sidecar didn't terminate, killing...")
                proc.kill()
                proc.wait(timeout=5.0)


def test_detector_connection_via_ulster37_sidecar(tmp_path: Path):
    """
    Test that we can connect to real Pixet detectors using ulster37 sidecar.
    This validates the legacy Python environment can communicate with Pixet SDK.
    """
    host = "127.0.0.1"
    sidecar_port = _free_tcp_port()
    
    config = _read_json(SETUP_CONFIG)
    config["DEV"] = False
    config["measurements_folder"] = str(tmp_path / "measurements")
    
    env_overrides = {
        "DETECTOR_BACKEND": "sidecar",
        "PIXET_SIDECAR_HOST": host,
        "PIXET_SIDECAR_PORT": str(sidecar_port),
    }
    
    print(f"\n[TEST] Starting ulster37 sidecar on port {sidecar_port}")
    
    with _started_sidecar_ulster37(host, sidecar_port):
        with _temporary_env(env_overrides):
            print("[TEST] Initializing DirectHardwareClient...")
            client = DirectHardwareClient(config)
            
            print("[TEST] Attempting to initialize detectors...")
            detector_init = client.initialize_detector()
            
            if not detector_init:
                pytest.fail("Failed to initialize detectors via ulster37 sidecar")
            
            print("[TEST] ✓ Detectors initialized successfully")
            
            # Try a quick capture to verify hardware is functional
            exposure_s = 0.2
            print(f"[TEST] Capturing test exposure ({exposure_s}s)...")
            
            try:
                outputs = client.capture_exposure(
                    exposure_s=exposure_s,
                    frames=1,
                    timeout_s=max(30.0, exposure_s + 20.0),
                )
                
                print(f"[TEST] ✓ Capture successful, got outputs: {list(outputs.keys())}")
                
                # Verify output files exist
                for alias, path in outputs.items():
                    output_path = Path(path)
                    assert output_path.exists(), f"Missing output for {alias}: {path}"
                    print(f"[TEST] ✓ Output file exists for {alias}: {output_path}")
                
            finally:
                client.deinitialize()


def test_stage_connection_via_eosdx13(tmp_path: Path):
    """
    Test that we can connect to the Kinesis XY stage.
    This runs in the current environment (should be eosdx13).
    """
    config = _read_json(SETUP_CONFIG)
    config["DEV"] = False
    config["measurements_folder"] = str(tmp_path / "measurements")
    
    # For stage-only test, we can use dummy detectors
    env_overrides = {
        "DETECTOR_BACKEND": "dummy",
    }
    
    with _temporary_env(env_overrides):
        print("\n[TEST] Initializing DirectHardwareClient for stage test...")
        client = DirectHardwareClient(config)
        
        # We need to init detectors even if dummy, to test stage independently
        print("[TEST] Initializing dummy detectors...")
        detector_init = client.initialize_detector()
        assert detector_init, "Failed to initialize dummy detectors"
        
        print("[TEST] Attempting to initialize motion stage...")
        motion_init = client.initialize_motion()
        
        if not motion_init:
            pytest.fail("Failed to initialize Kinesis stage")
        
        print("[TEST] ✓ Motion stage initialized successfully")
        
        # Test basic motion operations
        print("[TEST] Getting current position...")
        x, y = client.get_xy_position()
        print(f"[TEST] Current position: x={x:.3f}mm, y={y:.3f}mm")
        
        # Move to same position (safe operation)
        print(f"[TEST] Moving to current position (x={x:.3f}, y={y:.3f})...")
        moved_x, moved_y = client.move_to(x, y, timeout_s=20.0)
        
        print(f"[TEST] Moved to: x={moved_x:.3f}mm, y={moved_y:.3f}mm")
        assert moved_x == pytest.approx(x, abs=0.001), "X position mismatch"
        assert moved_y == pytest.approx(y, abs=0.001), "Y position mismatch"
        
        print("[TEST] ✓ Stage movement successful")
        
        client.deinitialize()


def test_full_system_with_split_envs(tmp_path: Path):
    """
    Full integration test: ulster37 sidecar for detectors + eosdx13 for stage.
    This is the most realistic test of the production hardware stack.
    """
    host = "127.0.0.1"
    sidecar_port = _free_tcp_port()
    
    config = _read_json(SETUP_CONFIG)
    config["DEV"] = False
    config["measurements_folder"] = str(tmp_path / "measurements")
    
    env_overrides = {
        "DETECTOR_BACKEND": "sidecar",
        "PIXET_SIDECAR_HOST": host,
        "PIXET_SIDECAR_PORT": str(sidecar_port),
    }
    
    print(f"\n[TEST] Full system test: ulster37 sidecar on port {sidecar_port}")
    
    with _started_sidecar_ulster37(host, sidecar_port):
        with _temporary_env(env_overrides):
            print("[TEST] Initializing full hardware stack...")
            client = DirectHardwareClient(config)
            
            print("[TEST] Initializing detectors (via ulster37 sidecar)...")
            detector_init = client.initialize_detector()
            assert detector_init, "Failed to initialize detectors"
            print("[TEST] ✓ Detectors initialized")
            
            print("[TEST] Initializing motion (via eosdx13)...")
            motion_init = client.initialize_motion()
            assert motion_init, "Failed to initialize motion"
            print("[TEST] ✓ Motion initialized")
            
            # Test coordinated operation
            print("[TEST] Getting current position...")
            x, y = client.get_xy_position()
            print(f"[TEST] Current position: x={x:.3f}mm, y={y:.3f}mm")
            
            print("[TEST] Moving to current position...")
            moved_x, moved_y = client.move_to(x, y, timeout_s=20.0)
            assert moved_x == pytest.approx(x, abs=0.001)
            assert moved_y == pytest.approx(y, abs=0.001)
            print("[TEST] ✓ Motion successful")
            
            print("[TEST] Capturing exposure...")
            exposure_s = 0.2
            outputs = client.capture_exposure(
                exposure_s=exposure_s,
                frames=1,
                timeout_s=max(30.0, exposure_s + 20.0),
            )
            print(f"[TEST] ✓ Capture successful: {list(outputs.keys())}")
            
            # Verify outputs
            for alias, path in outputs.items():
                output_path = Path(path)
                assert output_path.exists(), f"Missing output for {alias}: {path}"
            
            print("[TEST] ✓ Full system test PASSED")
            
            client.deinitialize()


if __name__ == "__main__":
    # Allow running directly for manual testing
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        print("=" * 80)
        print("Running split environment hardware tests")
        print("=" * 80)
        
        try:
            test_detector_connection_via_ulster37_sidecar(tmp_path)
            print("\n" + "=" * 80)
            test_stage_connection_via_eosdx13(tmp_path)
            print("\n" + "=" * 80)
            test_full_system_with_split_envs(tmp_path)
            print("\n" + "=" * 80)
            print("ALL TESTS PASSED ✓")
        except Exception as e:
            print(f"\nTEST FAILED: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)
