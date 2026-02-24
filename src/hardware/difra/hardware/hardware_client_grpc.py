from __future__ import annotations

import os
import re
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from hardware.difra.hardware.hardware_client_axis import normalize_axis
from hardware.difra.hardware.hardware_client_types import (
    CommandReadiness,
    HardwareClient,
)
from hardware.difra.hardware.hardware_control import HardwareController

try:
    import grpc
except Exception as exc:  # pragma: no cover - environment-dependent import
    grpc = None
    _GRPC_IMPORT_ERROR = exc
else:
    _GRPC_IMPORT_ERROR = None

try:
    from google.protobuf.timestamp_pb2 import Timestamp
except Exception as exc:  # pragma: no cover - environment-dependent import
    Timestamp = None
    _PROTOBUF_IMPORT_ERROR = exc
else:
    _PROTOBUF_IMPORT_ERROR = None

_GENERATED_STUB_ROOT = Path(__file__).resolve().parents[1] / "grpc_server" / "generated"
if str(_GENERATED_STUB_ROOT) not in sys.path:
    sys.path.insert(0, str(_GENERATED_STUB_ROOT))

if grpc is not None:
    try:
        from hub.v1 import hub_pb2, hub_pb2_grpc
    except Exception as exc:  # pragma: no cover - environment-dependent import
        hub_pb2 = None
        hub_pb2_grpc = None
        _HUB_IMPORT_ERROR = exc
    else:
        _HUB_IMPORT_ERROR = None
else:  # pragma: no cover - exercised when grpc is missing
    hub_pb2 = None
    hub_pb2_grpc = None
    _HUB_IMPORT_ERROR = _GRPC_IMPORT_ERROR

if grpc is not None:
    FALLBACK_GRPC_EXCEPTIONS = (
        grpc.RpcError,
        grpc.FutureTimeoutError,
        TimeoutError,
        OSError,
        RuntimeError,
    )
else:  # pragma: no cover - exercised when grpc is missing
    FALLBACK_GRPC_EXCEPTIONS = (
        TimeoutError,
        OSError,
        RuntimeError,
        ImportError,
        ModuleNotFoundError,
    )


def grpc_runtime_available() -> bool:
    return grpc is not None and Timestamp is not None and hub_pb2 is not None and hub_pb2_grpc is not None


def grpc_import_error() -> Optional[Exception]:
    return _GRPC_IMPORT_ERROR or _PROTOBUF_IMPORT_ERROR or _HUB_IMPORT_ERROR


def _timestamp_now() -> Timestamp:
    if Timestamp is None:
        raise RuntimeError("protobuf Timestamp is unavailable")
    ts = Timestamp()
    ts.FromDatetime(datetime.now(timezone.utc))
    return ts


def _command_context(user: str, reason: str) -> hub_pb2.CommandContext:
    if hub_pb2 is None:
        raise RuntimeError("gRPC protobuf stubs are unavailable")
    return hub_pb2.CommandContext(
        command_id=str(uuid.uuid4()),
        user=user,
        reason=reason,
        timestamp=_timestamp_now(),
        measurement_class=hub_pb2.SAMPLE,
    )


class GrpcHardwareClient(HardwareClient):
    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 50061,
        timeout_s: float = 3.0,
        user: str = "difra_gui",
    ):
        if not grpc_runtime_available():
            import_error = grpc_import_error()
            detail = f": {import_error}" if import_error else ""
            raise RuntimeError(
                f"grpcio/protobuf stubs unavailable; gRPC client disabled{detail}"
            )
        self._host = host
        self._port = int(port)
        self._timeout_s = float(timeout_s)
        self._user = user
        self._target = f"{self._host}:{self._port}"
        self._channel = grpc.insecure_channel(self._target)

        self._acquisition = hub_pb2_grpc.AcquisitionStub(self._channel)
        self._motion = hub_pb2_grpc.MotionStub(self._channel)
        self._device_init = hub_pb2_grpc.DeviceInitializationStub(self._channel)
        self._state_monitor = hub_pb2_grpc.StateMonitorStub(self._channel)
        self._discovery = hub_pb2_grpc.CommandDiscoveryStub(self._channel)

    def _wait_channel(self) -> None:
        grpc.channel_ready_future(self._channel).result(timeout=self._timeout_s)

    def initialize_detector(self) -> bool:
        self._wait_channel()
        response = self._device_init.InitializeDetector(
            hub_pb2.InitializeDetectorRequest(
                ctx=_command_context(self._user, "initialize_detector")
            ),
            timeout=self._timeout_s,
        )
        return bool(response.initialized)

    def initialize_motion(self) -> bool:
        self._wait_channel()
        response = self._device_init.InitializeMotion(
            hub_pb2.InitializeMotionRequest(
                ctx=_command_context(self._user, "initialize_motion")
            ),
            timeout=self._timeout_s,
        )
        return bool(response.initialized)

    def deinitialize(self) -> None:
        self._wait_channel()
        ctx = _command_context(self._user, "deinitialize")
        try:
            self._device_init.PowerOffDetector(ctx, timeout=self._timeout_s)
        except grpc.RpcError:
            pass
        try:
            self._device_init.PowerOffMotion(ctx, timeout=self._timeout_s)
        except grpc.RpcError:
            pass

    def move_to(
        self,
        position_mm: float,
        axis: Any,
        timeout_s: float = 25.0,
    ) -> Tuple[float, float]:
        self._wait_channel()
        axis_name = normalize_axis(axis)
        before_x, before_y = self.get_xy_position()
        target = float(position_mm)
        self._motion.MoveTo(
            hub_pb2.MoveToRequest(
                ctx=_command_context(self._user, f"move_to axis:{axis_name}"),
                position_mm=target,
            ),
            timeout=float(timeout_s),
        )
        after_x, after_y = self.get_xy_position()
        tolerance_mm = float(os.environ.get("DIFRA_MOVE_AXIS_TOL_MM", "0.05"))
        if axis_name == "x":
            if abs(after_x - target) > tolerance_mm:
                raise RuntimeError(
                    f"Motion.MoveTo(axis=x) target mismatch: requested x={target:.4f}, got x={after_x:.4f}"
                )
        else:
            if abs(after_y - target) > tolerance_mm:
                if (
                    abs(after_x - target) <= tolerance_mm
                    and abs(after_y - before_y) <= tolerance_mm
                ):
                    raise RuntimeError(
                        "Motion.MoveTo(axis=y) was applied to X axis. "
                        "Stale gRPC server detected; restart launcher to refresh sidecars."
                    )
                raise RuntimeError(
                    f"Motion.MoveTo(axis=y) target mismatch: requested y={target:.4f}, got y={after_y:.4f}"
                )
        return after_x, after_y

    def home(self, timeout_s: float = 25.0) -> Tuple[float, float]:
        self._wait_channel()
        self._motion.Home(
            hub_pb2.HomeRequest(ctx=_command_context(self._user, "home")),
            timeout=float(timeout_s),
        )
        return self.get_xy_position()

    def get_xy_position(self) -> Tuple[float, float]:
        self._wait_channel()
        motion_state = self._state_monitor.GetMotionState(
            hub_pb2.Empty(), timeout=self._timeout_s
        )
        x = (
            float(motion_state.position_x)
            if motion_state.HasField("position_x")
            else 0.0
        )
        y = (
            float(motion_state.position_y)
            if motion_state.HasField("position_y")
            else 0.0
        )
        return x, y

    def get_command_readiness(self) -> Dict[Tuple[str, str], CommandReadiness]:
        self._wait_channel()
        response = self._discovery.GetCommandReadiness(
            hub_pb2.Empty(), timeout=self._timeout_s
        )
        readiness: Dict[Tuple[str, str], CommandReadiness] = {}
        for item in response.items:
            readiness[(item.service_name, item.command_name)] = CommandReadiness(
                ready=bool(item.ready),
                reasons=list(item.reasons),
            )
        return readiness

    def get_state(self) -> Dict[str, Any]:
        self._wait_channel()
        response = self._acquisition.GetState(hub_pb2.Empty(), timeout=self._timeout_s)
        return {
            "state": int(response.state),
            "detail": response.detail,
            "mode": "grpc",
            "locks": {
                "device_locked": bool(response.locks.device_locked),
                "session_locked": bool(response.locks.session_locked),
                "technical_container_locked": bool(
                    response.locks.technical_container_locked
                ),
            },
        }

    def capture_exposure(
        self,
        exposure_s: float,
        frames: int = 1,
        timeout_s: float = 120.0,
    ) -> Dict[str, str]:
        self._wait_channel()
        total_ms = max(
            1,
            int(round(float(exposure_s) * max(int(frames), 1) * 1000.0)),
        )
        max_timeout_ms = max(total_ms + 5000, int(float(timeout_s) * 1000.0))
        self._acquisition.StartExposure(
            hub_pb2.StartExposureRequest(
                ctx=_command_context(self._user, "start_exposure"),
                exposure_time_ms=total_ms,
                max_timeout_ms=max_timeout_ms,
            ),
            timeout=float(timeout_s),
        )

        running_states = {
            hub_pb2.PENDING_ARMED,
            hub_pb2.RUNNING,
            hub_pb2.PAUSED,
            hub_pb2.STOPPING,
        }
        deadline = time.time() + max(float(timeout_s), float(total_ms) / 1000.0 + 10.0)
        while time.time() < deadline:
            state = self._acquisition.GetState(hub_pb2.Empty(), timeout=self._timeout_s)
            if int(state.state) not in running_states:
                break
            time.sleep(0.02)
        else:
            raise TimeoutError(f"Exposure did not complete within timeout {timeout_s}s")

        result = self._acquisition.GetLastExposureResult(
            hub_pb2.Empty(),
            timeout=self._timeout_s,
        )
        if not bool(result.has_result) or not result.result.data_path:
            raise RuntimeError("No exposure result was reported by gRPC server")

        result_path = Path(result.result.data_path)
        stem = result_path.stem
        match = re.match(r"^([0-9a-fA-F-]{36})_(.+)$", stem)
        if match:
            run_id = match.group(1)
            parent = result_path.parent
            if parent.exists():
                txt_files = sorted(parent.glob(f"{run_id}_*.txt"))
                if txt_files:
                    outputs: Dict[str, str] = {}
                    for txt_path in txt_files:
                        alias_tag = txt_path.stem[len(run_id) + 1 :]
                        outputs[alias_tag] = str(txt_path)
                    return outputs

        alias = match.group(2) if match else stem
        return {alias: str(result_path)}

    @property
    def stage_controller(self) -> Any:
        return None

    @property
    def detector_controllers(self) -> Dict[str, Any]:
        return {}

    @property
    def hardware_controller(self) -> Optional[HardwareController]:
        return None

    def close(self) -> None:
        self._channel.close()
