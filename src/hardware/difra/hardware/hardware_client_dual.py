from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Tuple

from hardware.difra.hardware.hardware_client_direct import DirectHardwareClient
from hardware.difra.hardware.hardware_client_grpc import (
    FALLBACK_GRPC_EXCEPTIONS,
    GrpcHardwareClient,
)
from hardware.difra.hardware.hardware_client_types import (
    CommandReadiness,
    HardwareClient,
)
from hardware.difra.hardware.hardware_control import HardwareController

LOGGER = logging.getLogger(__name__)


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
        fallback_on: Tuple[type, ...] = FALLBACK_GRPC_EXCEPTIONS + (NotImplementedError,),
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
            LOGGER.warning("HardwareClient fallback to direct for %s: %s", operation, exc)
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
        position_mm: float,
        axis: Any,
        timeout_s: float = 25.0,
    ) -> Tuple[float, float]:
        return self._call(
            "move_to",
            grpc_call=lambda: self._grpc.move_to(
                position_mm=position_mm, axis=axis, timeout_s=timeout_s
            ),
            direct_call=lambda: self._direct.move_to(
                position_mm=position_mm, axis=axis, timeout_s=timeout_s
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
