from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from hardware.difra.hardware.hardware_control import HardwareController


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
        position_mm: float,
        axis: Any,
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
