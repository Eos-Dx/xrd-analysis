from __future__ import annotations

import concurrent.futures
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from hardware.difra.hardware.hardware_client_axis import normalize_axis
from hardware.difra.hardware.hardware_client_types import (
    CommandReadiness,
    HardwareClient,
)
from hardware.difra.hardware.hardware_control import HardwareController


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
        position_mm: float,
        axis: Any,
        timeout_s: float = 25.0,
    ) -> Tuple[float, float]:
        if self.stage_controller is None:
            raise RuntimeError("Motion stage is not initialized")
        axis_name = normalize_axis(axis)
        current_x, current_y = self._controller.get_xy_position()
        if axis_name == "x":
            return self.stage_controller.move_stage(
                float(position_mm), float(current_y), move_timeout=timeout_s
            )
        return self.stage_controller.move_stage(
            float(current_x), float(position_mm), move_timeout=timeout_s
        )

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
                raise RuntimeError(f"No detector output produced for alias '{alias}'")
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
