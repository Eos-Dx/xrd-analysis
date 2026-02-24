"""Abstract detector controller contract."""

from abc import ABC, abstractmethod


class DetectorController(ABC):
    """Abstract base class for all detector controllers."""

    @abstractmethod
    def init_detector(self):
        pass

    @abstractmethod
    def capture_point(self, Nframes, Nseconds, filename_base):
        pass

    @abstractmethod
    def deinit_detector(self):
        pass

    @abstractmethod
    def start_stream(self, callback, exposure=0.1, interval=0.0, frames=1):
        pass

    @abstractmethod
    def stop_stream(self):
        pass

    @abstractmethod
    def convert_to_container_format(
        self,
        raw_file_path: str,
        container_version: str = "0.2",
    ) -> str:
        """Convert detector raw output to container format."""
        pass

    def get_raw_file_patterns(self):
        """Return list of glob patterns for detector raw output files."""
        return []
