from __future__ import annotations

import logging
import os
import re
import sys
import tempfile
import time
import uuid
import concurrent.futures
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

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

from hardware.difra.hardware.hardware_control import HardwareController

LOGGER = logging.getLogger(__name__)

_GENERATED_STUB_ROOT = (
    Path(__file__).resolve().parents[1] / "grpc_server" / "generated"
)
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
    _FALLBACK_GRPC_EXCEPTIONS = (
        grpc.RpcError,
        grpc.FutureTimeoutError,
        TimeoutError,
        OSError,
        RuntimeError,
    )
else:  # pragma: no cover - exercised when grpc is missing
    _FALLBACK_GRPC_EXCEPTIONS = (
        TimeoutError,
        OSError,
        RuntimeError,
        ImportError,
        ModuleNotFoundError,
    )


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


@dataclass
class CommandReadiness:
    ready: bool
    reasons: List[str]


class HardwareClient(ABC):
    @abstractmethod
    def initialize_detector(self) -> bool:
        pass

    @abstractmethod
    def initialize_motion(self) -> bool:
        pass

    @abstractmethod
    def deinitialize(self) -> None:
        pass

    @abstractmethod
    def move_to(
        self,
        x_mm: float,
        y_mm: Optional[float] = None,
        timeout_s: float = 25.0,
    ) -> Tuple[float, float]:
        pass

    @abstractmethod
    def home(self, timeout_s: float = 25.0) -> Tuple[float, float]:
        pass

    @abstractmethod
    def get_xy_position(self) -> Tuple[float, float]:
        pass

    @abstractmethod
    def get_command_readiness(self) -> Dict[Tuple[str, str], CommandReadiness]:
        pass

    @abstractmethod
    def get_state(self) -> Dict[str, Any]:
        pass

    @abstractmethod
    def capture_exposure(
        self,
        exposure_s: float,
        frames: int = 1,
        timeout_s: float = 120.0,
    ) -> Dict[str, str]:
        """Run detector exposure and return raw output paths keyed by detector alias."""
        pass

    @property
    @abstractmethod
    def stage_controller(self) -> Any:
        pass

    @property
    @abstractmethod
    def detector_controllers(self) -> Dict[str, Any]:
        pass

    @property
    @abstractmethod
    def hardware_controller(self) -> Optional[HardwareController]:
        pass


class DirectHardwareClient(HardwareClient):
    def __init__(self, config: Dict[str, Any]):
        self._config = config
        self._controller = HardwareController(config)
        self._motion_initialized = False
        self._detector_initialized = False

    def _initialize_components(
        self, init_motion: bool, init_detector: bool
    ) -> Tuple[bool, bool]:
        motion_ok, detector_ok = self._controller.initialize(
            init_stage=init_motion,
            init_detector=init_detector,
        )
        if init_motion:
            self._motion_initialized = bool(motion_ok)
        if init_detector:
            self._detector_initialized = bool(detector_ok)
        return self._motion_initialized, self._detector_initialized

    def initialize_detector(self) -> bool:
        _, detector_ok = self._initialize_components(
            init_motion=False,
            init_detector=True,
        )
        return detector_ok

    def initialize_motion(self) -> bool:
        motion_ok, _ = self._initialize_components(
            init_motion=True,
            init_detector=False,
        )
        return motion_ok

    def deinitialize(self) -> None:
        self._controller.deinitialize()
        self._motion_initialized = False
        self._detector_initialized = False

    def move_to(
        self,
        x_mm: float,
        y_mm: Optional[float] = None,
        timeout_s: float = 25.0,
    ) -> Tuple[float, float]:
        if self.stage_controller is None:
            raise RuntimeError("Motion stage is not initialized")
        if y_mm is None:
            _, y_mm = self._controller.get_xy_position()
        return self.stage_controller.move_stage(x_mm, y_mm, move_timeout=timeout_s)

    def home(self, timeout_s: float = 25.0) -> Tuple[float, float]:
        if self.stage_controller is None:
            raise RuntimeError("Motion stage is not initialized")
        return self.stage_controller.home_stage(timeout_s=timeout_s)

    def get_xy_position(self) -> Tuple[float, float]:
        return self._controller.get_xy_position()

    def get_command_readiness(self) -> Dict[Tuple[str, str], CommandReadiness]:
        running = False
        return {
            ("DeviceInitialization", "InitializeDetector"): CommandReadiness(
                ready=not self._detector_initialized,
                reasons=[]
                if not self._detector_initialized
                else ["Detector already initialized"],
            ),
            ("DeviceInitialization", "InitializeMotion"): CommandReadiness(
                ready=not self._motion_initialized,
                reasons=[]
                if not self._motion_initialized
                else ["Motion already initialized"],
            ),
            ("Acquisition", "GetState"): CommandReadiness(ready=True, reasons=[]),
            ("Motion", "MoveTo"): CommandReadiness(
                ready=self._motion_initialized,
                reasons=[]
                if self._motion_initialized
                else ["Motion stage is not initialized"],
            ),
            ("Motion", "Home"): CommandReadiness(
                ready=self._motion_initialized,
                reasons=[]
                if self._motion_initialized
                else ["Motion stage is not initialized"],
            ),
            ("Acquisition", "StartExposure"): CommandReadiness(
                ready=self._detector_initialized,
                reasons=[]
                if self._detector_initialized
                else ["Detector is not initialized"],
            ),
            ("Acquisition", "Pause"): CommandReadiness(
                ready=running,
                reasons=[] if running else ["No active exposure"],
            ),
            ("Acquisition", "Resume"): CommandReadiness(
                ready=False,
                reasons=["Exposure is not paused"],
            ),
            ("Acquisition", "Stop"): CommandReadiness(
                ready=running,
                reasons=[] if running else ["No active exposure"],
            ),
            ("Acquisition", "Abort"): CommandReadiness(
                ready=running,
                reasons=[] if running else ["No active exposure"],
            ),
        }

    def get_state(self) -> Dict[str, Any]:
        return {
            "motion_initialized": self._motion_initialized,
            "detector_initialized": self._detector_initialized,
            "mode": "direct",
            "locks": {
                "device_locked": False,
                "session_locked": False,
                "technical_container_locked": False,
            },
        }

    def capture_exposure(
        self,
        exposure_s: float,
        frames: int = 1,
        timeout_s: float = 120.0,
    ) -> Dict[str, str]:
        if not self.detector_controllers:
            raise RuntimeError("Detector is not initialized")

        out_dir = Path(tempfile.mkdtemp(prefix="difra_direct_capture_"))
        nframes = max(int(frames), 1)
        nseconds = float(exposure_s)

        def _capture_single(alias: str, controller: Any) -> Tuple[str, str]:
            base = out_dir / str(alias).replace(" ", "_")
            ok = bool(
                controller.capture_point(
                    Nframes=nframes,
                    Nseconds=nseconds,
                    filename_base=str(base),
                )
            )
            if not ok:
                raise RuntimeError(f"Capture failed for detector '{alias}'")

            txt_path = base.with_suffix(".txt")
            if txt_path.exists():
                return str(alias), str(txt_path)

            candidates = sorted(out_dir.glob(f"{base.name}.*"))
            if not candidates:
                raise RuntimeError(
                    f"No detector output produced for alias '{alias}'"
                )
            return str(alias), str(candidates[0])

        outputs: Dict[str, str] = {}
        max_workers = max(1, len(self.detector_controllers))
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = [
                pool.submit(_capture_single, alias, controller)
                for alias, controller in self.detector_controllers.items()
            ]
            for fut in concurrent.futures.as_completed(futures):
                alias, path = fut.result()
                outputs[alias] = path
        return outputs

    @property
    def stage_controller(self) -> Any:
        return self._controller.stage_controller

    @property
    def detector_controllers(self) -> Dict[str, Any]:
        return dict(self._controller.detectors)

    @property
    def hardware_controller(self) -> Optional[HardwareController]:
        return self._controller


class GrpcHardwareClient(HardwareClient):
    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 50061,
        timeout_s: float = 3.0,
        user: str = "difra_gui",
    ):
        if grpc is None or Timestamp is None or hub_pb2 is None or hub_pb2_grpc is None:
            import_error = (
                _GRPC_IMPORT_ERROR
                or _PROTOBUF_IMPORT_ERROR
                or _HUB_IMPORT_ERROR
            )
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
        x_mm: float,
        y_mm: Optional[float] = None,
        timeout_s: float = 25.0,
    ) -> Tuple[float, float]:
        self._wait_channel()
        _, current_y = self.get_xy_position()
        if y_mm is not None and abs(y_mm - current_y) > 1e-6:
            raise NotImplementedError(
                "Protocol v1 MoveTo is single-axis; y-axis move requires direct fallback"
            )
        self._motion.MoveTo(
            hub_pb2.MoveToRequest(
                ctx=_command_context(self._user, "move_to"),
                position_mm=float(x_mm),
            ),
            timeout=float(timeout_s),
        )
        return self.get_xy_position()

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
        x = float(motion_state.position_x) if motion_state.HasField("position_x") else 0.0
        y = float(motion_state.position_y) if motion_state.HasField("position_y") else 0.0
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
                "technical_container_locked": bool(response.locks.technical_container_locked),
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
            # Tighter poll interval keeps end-to-end exposure timing near requested duration.
            time.sleep(0.02)
        else:
            raise TimeoutError(
                f"Exposure did not complete within timeout {timeout_s}s"
            )

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


class DualPathHardwareClient(HardwareClient):
    def __init__(
        self,
        direct_client: DirectHardwareClient,
        grpc_client: Optional[GrpcHardwareClient],
        mode: str = "dual",
        sync_direct_detectors: bool = False,
    ):
        self._direct = direct_client
        self._grpc = grpc_client
        self._mode = mode
        self._sync_direct_detectors = bool(sync_direct_detectors)
        self.last_backend = "direct"
        self.last_fallback_reason = ""

    def _sync_direct_detectors_if_needed(self) -> None:
        if not self._sync_direct_detectors:
            return
        if self._direct.detector_controllers:
            return
        try:
            ok = self._direct.initialize_detector()
            if not ok:
                LOGGER.warning(
                    "Direct detector mirror initialization did not succeed while gRPC detector is active"
                )
        except Exception as exc:
            LOGGER.warning(
                "Direct detector mirror initialization failed while gRPC detector is active: %s",
                exc,
            )

    def _call(
        self,
        operation: str,
        grpc_call,
        direct_call,
        fallback_on: Tuple[type, ...] = _FALLBACK_GRPC_EXCEPTIONS
        + (NotImplementedError,),
    ):
        if self._mode == "direct" or self._grpc is None:
            self.last_backend = "direct"
            return direct_call()

        if self._mode == "grpc":
            self.last_backend = "grpc"
            return grpc_call()

        try:
            value = grpc_call()
            self.last_backend = "grpc"
            self.last_fallback_reason = ""
            return value
        except fallback_on as exc:
            self.last_backend = "direct"
            self.last_fallback_reason = str(exc)
            LOGGER.warning(
                "HardwareClient fallback to direct for %s: %s", operation, exc
            )
            return direct_call()

    def initialize_detector(self) -> bool:
        result = self._call(
            "initialize_detector",
            grpc_call=lambda: self._grpc.initialize_detector(),
            direct_call=self._direct.initialize_detector,
        )
        if self.last_backend == "grpc" and result:
            self._sync_direct_detectors_if_needed()
        return result

    def initialize_motion(self) -> bool:
        return self._call(
            "initialize_motion",
            grpc_call=lambda: self._grpc.initialize_motion(),
            direct_call=self._direct.initialize_motion,
        )

    def deinitialize(self) -> None:
        def _direct() -> None:
            self._direct.deinitialize()

        def _grpc() -> None:
            self._grpc.deinitialize()

        self._call("deinitialize", grpc_call=_grpc, direct_call=_direct)

    def move_to(
        self,
        x_mm: float,
        y_mm: Optional[float] = None,
        timeout_s: float = 25.0,
    ) -> Tuple[float, float]:
        return self._call(
            "move_to",
            grpc_call=lambda: self._grpc.move_to(
                x_mm=x_mm, y_mm=y_mm, timeout_s=timeout_s
            ),
            direct_call=lambda: self._direct.move_to(
                x_mm=x_mm, y_mm=y_mm, timeout_s=timeout_s
            ),
        )

    def home(self, timeout_s: float = 25.0) -> Tuple[float, float]:
        return self._call(
            "home",
            grpc_call=lambda: self._grpc.home(timeout_s=timeout_s),
            direct_call=lambda: self._direct.home(timeout_s=timeout_s),
        )

    def get_xy_position(self) -> Tuple[float, float]:
        return self._call(
            "get_xy_position",
            grpc_call=self._grpc.get_xy_position,
            direct_call=self._direct.get_xy_position,
        )

    def get_command_readiness(self) -> Dict[Tuple[str, str], CommandReadiness]:
        return self._call(
            "get_command_readiness",
            grpc_call=self._grpc.get_command_readiness,
            direct_call=self._direct.get_command_readiness,
        )

    def get_state(self) -> Dict[str, Any]:
        return self._call(
            "get_state",
            grpc_call=self._grpc.get_state,
            direct_call=self._direct.get_state,
        )

    def _normalize_capture_outputs(self, outputs: Dict[str, str]) -> Dict[str, str]:
        if not outputs:
            return outputs
        aliases = list(self._direct.detector_controllers.keys())
        if not aliases:
            return outputs
        by_tag = {str(alias).replace(" ", "_"): str(alias) for alias in aliases}
        normalized: Dict[str, str] = {}
        for key, value in outputs.items():
            normalized[by_tag.get(str(key), str(key))] = value
        return normalized

    def capture_exposure(
        self,
        exposure_s: float,
        frames: int = 1,
        timeout_s: float = 120.0,
    ) -> Dict[str, str]:
        outputs = self._call(
            "capture_exposure",
            grpc_call=lambda: self._grpc.capture_exposure(
                exposure_s=exposure_s,
                frames=frames,
                timeout_s=timeout_s,
            ),
            direct_call=lambda: self._direct.capture_exposure(
                exposure_s=exposure_s,
                frames=frames,
                timeout_s=timeout_s,
            ),
        )
        return self._normalize_capture_outputs(outputs)

    @property
    def stage_controller(self) -> Any:
        return self._direct.stage_controller

    @property
    def detector_controllers(self) -> Dict[str, Any]:
        return self._direct.detector_controllers

    @property
    def hardware_controller(self) -> Optional[HardwareController]:
        return self._direct.hardware_controller


def create_hardware_client(config: Dict[str, Any]) -> HardwareClient:
    protocol_cfg = (config or {}).get("hardware_protocol", {})
    detector_backend = str(os.environ.get("DETECTOR_BACKEND", "")).lower().strip()
    default_mode = "dual"
    mode_override = str(
        os.environ.get("HARDWARE_CLIENT_MODE")
        or os.environ.get("DIFRA_HARDWARE_CLIENT_MODE")
        or ""
    ).lower().strip()
    mode = str(mode_override or protocol_cfg.get("client_mode", default_mode)).lower().strip()
    if mode not in {"dual", "direct", "grpc"}:
        mode = default_mode

    sync_direct_detectors_cfg = protocol_cfg.get("sync_direct_detectors")
    if sync_direct_detectors_cfg is None:
        sync_direct_detectors = detector_backend in {"sidecar", "socket", "ipc"}
    else:
        sync_direct_detectors = bool(sync_direct_detectors_cfg)

    direct_client = DirectHardwareClient(config)

    grpc_client: Optional[GrpcHardwareClient] = None
    if mode in {"dual", "grpc"}:
        if grpc is None or Timestamp is None or hub_pb2 is None or hub_pb2_grpc is None:
            import_error = (
                _GRPC_IMPORT_ERROR
                or _PROTOBUF_IMPORT_ERROR
                or _HUB_IMPORT_ERROR
            )
            msg = (
                f"grpcio/protobuf stubs unavailable in this environment"
                + (f": {import_error}" if import_error else "")
            )
            if mode == "grpc":
                raise RuntimeError(msg)
            LOGGER.warning("%s; falling back to direct mode", msg)
            mode = "direct"
            return DualPathHardwareClient(
                direct_client=direct_client,
                grpc_client=None,
                mode=mode,
                sync_direct_detectors=sync_direct_detectors,
            )

        host = str(
            os.environ.get("DIFRA_GRPC_HOST")
            or protocol_cfg.get("grpc_host", "127.0.0.1")
        )
        port = int(
            os.environ.get("DIFRA_GRPC_PORT") or protocol_cfg.get("grpc_port", 50061)
        )
        timeout_s = float(
            os.environ.get("DIFRA_GRPC_TIMEOUT_S")
            or protocol_cfg.get("grpc_timeout_s", 3.0)
        )
        user = str(
            os.environ.get("DIFRA_GRPC_USER")
            or protocol_cfg.get("grpc_user", "difra_gui")
        )

        try:
            grpc_client = GrpcHardwareClient(
                host=host,
                port=port,
                timeout_s=timeout_s,
                user=user,
            )
        except Exception as exc:
            if mode == "grpc":
                raise
            LOGGER.warning(
                "Failed to initialize gRPC hardware client, using direct mode: %s",
                exc,
            )
            mode = "direct"

    return DualPathHardwareClient(
        direct_client=direct_client,
        grpc_client=grpc_client,
        mode=mode,
        sync_direct_detectors=sync_direct_detectors,
    )
