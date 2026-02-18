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
from typing import Dict, Iterator, List

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
GLOBAL_CONFIG = REPO_ROOT / "src" / "hardware" / "difra" / "resources" / "config" / "global.json"
MAIN_CONFIG = REPO_ROOT / "src" / "hardware" / "difra" / "resources" / "config" / "main.json"


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


def _read_json(path: Path) -> Dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _resolve_setup_config_path() -> Path:
    override = os.environ.get("DIFRA_REAL_SETUP_CONFIG", "").strip()
    if override:
        path = Path(override).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"DIFRA_REAL_SETUP_CONFIG does not exist: {path}")
        return path

    chosen = MAIN_CONFIG
    if GLOBAL_CONFIG.exists():
        try:
            global_cfg = _read_json(GLOBAL_CONFIG)
            setup = str(global_cfg.get("default_setup", "")).strip()
            if setup:
                setup_path = GLOBAL_CONFIG.parent / "setups" / f"{setup}.json"
                if setup_path.exists():
                    chosen = setup_path
        except Exception:
            pass
    return chosen


def _active_detector_aliases(config: Dict) -> List[str]:
    detector_cfgs = list(config.get("detectors", []) or [])
    active_ids = set(config.get("active_detectors", []) or [])
    aliases: List[str] = []
    for det in detector_cfgs:
        if det.get("id") in active_ids:
            alias = str(det.get("alias", "")).strip()
            if alias:
                aliases.append(alias)
    return aliases


def _active_stage_type(config: Dict) -> str:
    stages = list(config.get("translation_stages", []) or [])
    active_ids = set(config.get("active_translation_stages", []) or [])
    for stage in stages:
        if stage.get("id") in active_ids:
            return str(stage.get("type", "")).strip()
    return str(stages[0].get("type", "")).strip() if stages else ""


def _assert_expected_hardware_route(
    config: Dict,
    stage_controller: object,
    detector_controllers: Dict[str, object],
) -> None:
    expected_stage_type = os.environ.get("DIFRA_EXPECT_STAGE_TYPE", "Kinesis").strip()
    if expected_stage_type:
        active_stage_type = _active_stage_type(config)
        assert (
            active_stage_type == expected_stage_type
        ), f"Expected active stage type '{expected_stage_type}', got '{active_stage_type}'"

    expected_stage_class = os.environ.get(
        "DIFRA_EXPECT_STAGE_CLASS", "XYStageLibController"
    ).strip()
    if expected_stage_class:
        actual_stage_class = type(stage_controller).__name__
        assert (
            actual_stage_class == expected_stage_class
        ), f"Expected stage controller '{expected_stage_class}', got '{actual_stage_class}'"

    expected_detector_class = os.environ.get(
        "DIFRA_EXPECT_DETECTOR_CLASS", "PixetSidecarDetectorController"
    ).strip()
    if expected_detector_class:
        assert detector_controllers, "No detector controllers initialized"
        bad = {
            alias: type(ctrl).__name__
            for alias, ctrl in detector_controllers.items()
            if type(ctrl).__name__ != expected_detector_class
        }
        assert not bad, f"Detector route mismatch (expected {expected_detector_class}): {bad}"


def _active_stage_limits(config: Dict) -> Dict[str, tuple[float, float]]:
    stages = list(config.get("translation_stages", []) or [])
    active_ids = set(config.get("active_translation_stages", []) or [])
    selected = None
    for stage in stages:
        if stage.get("id") in active_ids:
            selected = stage
            break
    if selected is None and stages:
        selected = stages[0]
    if selected is None:
        raise RuntimeError("No translation stage configured in selected setup config")

    limits_cfg = (selected.get("settings", {}) or {}).get("limits_mm", {}) or {}
    try:
        x_limits = tuple(float(v) for v in limits_cfg.get("x", (-14.0, 14.0)))
        y_limits = tuple(float(v) for v in limits_cfg.get("y", (-14.0, 14.0)))
        if len(x_limits) != 2 or len(y_limits) != 2:
            raise ValueError("Invalid limits shape")
    except Exception as exc:
        raise RuntimeError(f"Invalid stage limits in config: {exc}") from exc
    return {"x": (x_limits[0], x_limits[1]), "y": (y_limits[0], y_limits[1])}


def _safe_axis_target(current: float, limits: tuple[float, float], delta: float) -> float:
    lo, hi = float(limits[0]), float(limits[1])
    candidate = float(current) + float(delta)
    if lo <= candidate <= hi:
        return candidate
    candidate = float(current) - float(delta)
    if lo <= candidate <= hi:
        return candidate
    margin = 0.1
    return min(max(float(current), lo + margin), hi - margin)


def _free_tcp_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


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
        deadline = time.time() + 30.0
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
                proc.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=5.0)


def _real_config(tmp_path: Path) -> Dict:
    cfg = _read_json(_resolve_setup_config_path())
    cfg["DEV"] = False
    cfg["measurements_folder"] = str(tmp_path / "measurements")
    return cfg


def test_real_hardware_direct_legacy_sidecar_smoke(tmp_path: Path):
    host = "127.0.0.1"
    sidecar_port = _free_tcp_port()
    config = _real_config(tmp_path)
    expected_aliases = _active_detector_aliases(config)
    if not expected_aliases:
        raise RuntimeError("No active real detectors in selected setup config")

    stage_delta_mm = float(os.environ.get("DIFRA_REAL_HW_STAGE_DELTA_MM", "0.5"))
    stage_limits = _active_stage_limits(config)
    exposure_s = float(os.environ.get("DIFRA_REAL_HW_EXPOSURE_S", "0.2"))
    env = {
        "DETECTOR_BACKEND": "sidecar",
        "PIXET_SIDECAR_HOST": host,
        "PIXET_SIDECAR_PORT": str(sidecar_port),
    }

    with _started_sidecar(host, sidecar_port):
        with _temporary_env(env):
            client = DirectHardwareClient(config)
            try:
                assert client.initialize_detector() is True, (
                    "Detector init via legacy sidecar failed. "
                    "Set DIFRA_LEGACY_ENV or DIFRA_LEGACY_PYTHON to a working legacy runtime."
                )
                assert client.initialize_motion() is True
                _assert_expected_hardware_route(
                    config=config,
                    stage_controller=client.stage_controller,
                    detector_controllers=client.detector_controllers,
                )

                start_x, start_y = client.get_xy_position()
                target_x = _safe_axis_target(start_x, stage_limits["x"], stage_delta_mm)
                target_y = _safe_axis_target(start_y, stage_limits["y"], stage_delta_mm)

                moved_x, moved_y = client.move_to(target_x, axis="x", timeout_s=20.0)
                assert moved_x == pytest.approx(target_x, abs=1e-3)
                moved_x, moved_y = client.move_to(target_y, axis="y", timeout_s=20.0)
                assert moved_y == pytest.approx(target_y, abs=1e-3)

                outputs = client.capture_exposure(
                    exposure_s=exposure_s,
                    frames=1,
                    timeout_s=max(30.0, exposure_s + 20.0),
                )

                # Return to initial position to keep physical state stable for subsequent tests.
                client.move_to(start_x, axis="x", timeout_s=20.0)
                client.move_to(start_y, axis="y", timeout_s=20.0)
            finally:
                client.deinitialize()

    assert set(expected_aliases).issubset(set(outputs.keys()))
    for alias in expected_aliases:
        path = Path(outputs[alias])
        assert path.exists(), f"Missing output for {alias}: {path}"


def test_real_hardware_grpc_over_legacy_sidecar_smoke(tmp_path: Path):
    async def _scenario() -> Dict[str, str]:
        host = "127.0.0.1"
        sidecar_port = _free_tcp_port()
        config = _real_config(tmp_path)
        stage_limits = _active_stage_limits(config)
        stage_delta_mm = float(os.environ.get("DIFRA_REAL_HW_STAGE_DELTA_MM", "0.5"))
        expected_aliases = _active_detector_aliases(config)
        if not expected_aliases:
            raise RuntimeError("No active real detectors in selected setup config")
        exposure_s = float(os.environ.get("DIFRA_REAL_HW_EXPOSURE_S", "0.2"))
        exposure_ms = max(1, int(round(exposure_s * 1000.0)))

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
                    motion_stub = hub_pb2_grpc.MotionStub(channel)

                    init_detector = await init_stub.InitializeDetector(
                        hub_pb2.InitializeDetectorRequest(ctx=_ctx("init_detector_real"))
                    )
                    init_motion = await init_stub.InitializeMotion(
                        hub_pb2.InitializeMotionRequest(ctx=_ctx("init_motion_real"))
                    )
                    assert init_detector.initialized is True, (
                        "Detector init via legacy sidecar failed. "
                        "Set DIFRA_LEGACY_ENV or DIFRA_LEGACY_PYTHON to a working legacy runtime."
                    )
                    assert init_motion.initialized is True
                    _assert_expected_hardware_route(
                        config=config,
                        stage_controller=server.state.stage_controller,
                        detector_controllers=server.state.detector_controllers,
                    )

                    state_stub = hub_pb2_grpc.StateMonitorStub(channel)
                    motion_state = await state_stub.GetMotionState(hub_pb2.Empty())
                    start_x = float(motion_state.position_x)
                    start_y = float(motion_state.position_y)
                    target_x = _safe_axis_target(start_x, stage_limits["x"], stage_delta_mm)
                    target_y = _safe_axis_target(start_y, stage_limits["y"], stage_delta_mm)

                    await motion_stub.MoveTo(
                        hub_pb2.MoveToRequest(
                            ctx=_ctx("move_to_real_x axis:x"),
                            position_mm=target_x,
                        )
                    )
                    await motion_stub.MoveTo(
                        hub_pb2.MoveToRequest(
                            ctx=_ctx("move_to_real_y axis:y"),
                            position_mm=target_y,
                        )
                    )
                    motion_state = await state_stub.GetMotionState(hub_pb2.Empty())
                    assert float(motion_state.position_x) == pytest.approx(target_x, abs=1e-3)
                    assert float(motion_state.position_y) == pytest.approx(target_y, abs=1e-3)

                    await acq_stub.StartExposure(
                        hub_pb2.StartExposureRequest(
                            ctx=_ctx("start_exposure_real"),
                            exposure_time_ms=exposure_ms,
                            max_timeout_ms=max(exposure_ms + 20000, 30000),
                        )
                    )
                    running_states = {
                        hub_pb2.PENDING_ARMED,
                        hub_pb2.RUNNING,
                        hub_pb2.PAUSED,
                        hub_pb2.STOPPING,
                    }
                    deadline = time.time() + 120.0
                    while time.time() < deadline:
                        state = await acq_stub.GetState(hub_pb2.Empty())
                        if int(state.state) not in running_states:
                            break
                        await asyncio.sleep(0.1)
                    else:
                        raise TimeoutError("Real hardware exposure did not complete in time")

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

                    await motion_stub.MoveTo(
                        hub_pb2.MoveToRequest(
                            ctx=_ctx("restore_real_x axis:x"),
                            position_mm=start_x,
                        )
                    )
                    await motion_stub.MoveTo(
                        hub_pb2.MoveToRequest(
                            ctx=_ctx("restore_real_y axis:y"),
                            position_mm=start_y,
                        )
                    )
                    await channel.close()
                finally:
                    await server.stop(0)

        assert set(expected_aliases).issubset(set(outputs.keys()))
        return outputs

    outputs = asyncio.run(_scenario())
    assert outputs
