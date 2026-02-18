from __future__ import annotations

import asyncio
import contextlib
import json
import os
import re
import shutil
import socket
import subprocess
import sys
import time
import uuid
from pathlib import Path
from typing import Dict, Iterator

import grpc
import pytest
from google.protobuf.timestamp_pb2 import Timestamp

SRC_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if SRC_ROOT not in sys.path:
    sys.path.insert(0, SRC_ROOT)

from hardware.difra.grpc_server.server import DifraGrpcServer, hub_pb2, hub_pb2_grpc
from hardware.difra.hardware.hardware_client import DirectHardwareClient

REPO_ROOT = Path(__file__).resolve().parents[4]
SIDECAR_SCRIPT = REPO_ROOT / "src" / "hardware" / "difra" / "scripts" / "pixet_sidecar_server.py"


def _list_conda_env_names() -> set[str]:
    conda_exe = shutil.which("conda") or os.environ.get("CONDA_EXE", "").strip()
    if not conda_exe:
        return set()
    try:
        result = subprocess.run(
            [conda_exe, "env", "list", "--json"],
            capture_output=True,
            text=True,
            check=True,
        )
        payload = json.loads(result.stdout or "{}")
        env_paths = payload.get("envs", []) or []
        names = {Path(p).name for p in env_paths if p}
        return {n for n in names if n}
    except Exception:
        return set()


def _resolve_legacy_sidecar_command(host: str, port: int) -> list[str]:
    legacy_python = os.environ.get("DIFRA_LEGACY_PYTHON", "").strip()
    if legacy_python:
        return [
            legacy_python,
            "-u",
            str(SIDECAR_SCRIPT),
            "--host",
            host,
            "--port",
            str(port),
        ]

    conda_exe = shutil.which("conda") or os.environ.get("CONDA_EXE", "").strip()
    requested_env = os.environ.get("DIFRA_LEGACY_ENV", "").strip()
    if requested_env and conda_exe:
        return [
            conda_exe,
            "run",
            "--live-stream",
            "--no-capture-output",
            "-n",
            requested_env,
            "python",
            "-u",
            str(SIDECAR_SCRIPT),
            "--host",
            host,
            "--port",
            str(port),
        ]

    candidate_envs = [requested_env] if requested_env else ["ulster37"]
    available_envs = _list_conda_env_names()
    chosen_env = next((name for name in candidate_envs if name in available_envs), "")
    if chosen_env:
        try:
            version_check = subprocess.run(
                [
                    conda_exe,
                    "run",
                    "--no-capture-output",
                    "-n",
                    chosen_env,
                    "python",
                    "-c",
                    "import sys; print(f'{sys.version_info[0]}.{sys.version_info[1]}')",
                ],
                capture_output=True,
                text=True,
                check=True,
            )
            py_ver = (version_check.stdout or "").strip().splitlines()[-1]
            if py_ver != "3.7":
                raise RuntimeError(
                    f"Legacy env '{chosen_env}' must be Python 3.7, found '{py_ver}'"
                )
        except Exception as exc:
            raise RuntimeError(
                "Legacy sidecar runtime validation failed. "
                "Set DIFRA_LEGACY_ENV=ulster37 or DIFRA_LEGACY_PYTHON to Python 3.7."
            ) from exc
        return [
            conda_exe,
            "run",
            "--live-stream",
            "--no-capture-output",
            "-n",
            chosen_env,
            "python",
            "-u",
            str(SIDECAR_SCRIPT),
            "--host",
            host,
            "--port",
            str(port),
        ]

    raise RuntimeError(
        "No valid legacy sidecar runtime found. "
        "Set DIFRA_LEGACY_ENV=ulster37 or DIFRA_LEGACY_PYTHON to Python 3.7."
    )


def _free_tcp_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _dummy_dual_detector_config() -> Dict:
    return {
        "DEV": True,
        "detectors": [
            {
                "alias": "PRIMARY",
                "type": "DummyDetector",
                "id": "DUMMY-DET-1",
                "size": {"width": 16, "height": 16},
            },
            {
                "alias": "SECONDARY",
                "type": "DummyDetector",
                "id": "DUMMY-DET-2",
                "size": {"width": 16, "height": 16},
            },
        ],
        "dev_active_detectors": ["DUMMY-DET-1", "DUMMY-DET-2"],
        "active_detectors": [],
        "translation_stages": [],
        "dev_active_stages": [],
        "active_translation_stages": [],
    }


@contextlib.contextmanager
def _temporary_env(overrides: Dict[str, str]) -> Iterator[None]:
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


def _ctx(reason: str) -> hub_pb2.CommandContext:
    ts = Timestamp()
    ts.GetCurrentTime()
    return hub_pb2.CommandContext(
        command_id=str(uuid.uuid4()),
        user="pytest",
        reason=reason,
        timestamp=ts,
        measurement_class=hub_pb2.SAMPLE,
    )


def _sidecar_ping(host: str, port: int, timeout_s: float = 0.5) -> bool:
    payload = {"id": "ping", "cmd": "ping", "args": {}}
    raw = (json.dumps(payload) + "\n").encode("utf-8")
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


@contextlib.contextmanager
def _started_sidecar(host: str, port: int) -> Iterator[subprocess.Popen]:
    env = dict(os.environ)
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = f"{SRC_ROOT}{os.pathsep}{existing}" if existing else SRC_ROOT
    sidecar_cmd = _resolve_legacy_sidecar_command(host, port)
    proc = subprocess.Popen(
        sidecar_cmd,
        cwd=str(REPO_ROOT),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    try:
        deadline = time.time() + 10.0
        while time.time() < deadline:
            if proc.poll() is not None:
                out = proc.stdout.read() if proc.stdout else ""
                raise RuntimeError(f"Sidecar exited early with code {proc.returncode}: {out}")
            try:
                if _sidecar_ping(host, port):
                    break
            except OSError:
                time.sleep(0.1)
        else:
            raise TimeoutError("Timed out waiting for sidecar readiness")
        yield proc
    finally:
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=3.0)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=3.0)


def test_demo_integration_time_legacy_sidecar_path_is_parallel():
    host = "127.0.0.1"
    sidecar_port = _free_tcp_port()
    config = _dummy_dual_detector_config()
    env = {
        "DETECTOR_BACKEND": "sidecar",
        "PIXET_SIDECAR_HOST": host,
        "PIXET_SIDECAR_PORT": str(sidecar_port),
    }

    with _started_sidecar(host, sidecar_port):
        with _temporary_env(env):
            client = DirectHardwareClient(config)
            assert client.initialize_detector() is True
            t0 = time.perf_counter()
            outputs = client.capture_exposure(exposure_s=1.0, frames=1, timeout_s=10.0)
            elapsed = time.perf_counter() - t0
            client.deinitialize()

    assert set(outputs.keys()) == {"PRIMARY", "SECONDARY"}
    # 1s integration for two detectors must be truly parallel with minimal overhead.
    assert elapsed < 1.1


def test_demo_integration_time_grpc_path_is_parallel():
    async def _scenario() -> float:
        host = "127.0.0.1"
        sidecar_port = _free_tcp_port()
        config = _dummy_dual_detector_config()
        env = {
            "DETECTOR_BACKEND": "sidecar",
            "PIXET_SIDECAR_HOST": host,
            "PIXET_SIDECAR_PORT": str(sidecar_port),
        }

        with _started_sidecar(host, sidecar_port):
            with _temporary_env(env):
                server = DifraGrpcServer(config=config, host=host, port=0)
                await server.start()
                try:
                    channel = grpc.aio.insecure_channel(f"{host}:{server.bound_port}")
                    await channel.channel_ready()
                    init_stub = hub_pb2_grpc.DeviceInitializationStub(channel)
                    acq_stub = hub_pb2_grpc.AcquisitionStub(channel)

                    init_resp = await init_stub.InitializeDetector(
                        hub_pb2.InitializeDetectorRequest(ctx=_ctx("init_detector"))
                    )
                    assert init_resp.initialized is True

                    t0 = time.perf_counter()
                    await acq_stub.StartExposure(
                        hub_pb2.StartExposureRequest(
                            ctx=_ctx("start_exposure"),
                            exposure_time_ms=1000,
                            max_timeout_ms=12000,
                        )
                    )

                    running_states = {
                        hub_pb2.PENDING_ARMED,
                        hub_pb2.RUNNING,
                        hub_pb2.PAUSED,
                        hub_pb2.STOPPING,
                    }
                    deadline = time.time() + 15.0
                    while time.time() < deadline:
                        state = await acq_stub.GetState(hub_pb2.Empty())
                        if int(state.state) not in running_states:
                            break
                        await asyncio.sleep(0.01)
                    else:
                        raise TimeoutError("gRPC exposure did not complete in time")

                    # Measure integration runtime only; post-readout is validated separately.
                    elapsed = time.perf_counter() - t0

                    exposure = await acq_stub.GetLastExposureResult(hub_pb2.Empty())
                    assert bool(exposure.has_result) is True
                    assert bool(exposure.result.data_path) is True

                    first_path = Path(exposure.result.data_path)
                    stem = first_path.stem
                    match = re.match(r"^([0-9a-fA-F-]{36})_(.+)$", stem)
                    assert match is not None
                    run_id = match.group(1)
                    txt_files = sorted(first_path.parent.glob(f"{run_id}_*.txt"))
                    outputs = {
                        p.stem[len(run_id) + 1 :]: str(p)
                        for p in txt_files
                    }
                    await channel.close()
                finally:
                    await server.stop(0)

        assert set(outputs.keys()) == {"PRIMARY", "SECONDARY"}
        return elapsed

    elapsed = asyncio.run(_scenario())
    # 1s integration for two detectors must be truly parallel with minimal overhead.
    assert elapsed < 1.1
