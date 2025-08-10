from pathlib import Path
from typing import Dict

from PyQt5.QtCore import QObject, pyqtSignal


def validate_folder(path: str) -> str:
    """Ensure the target folder exists; return its absolute path."""
    p = Path(path).expanduser().resolve()
    p.mkdir(parents=True, exist_ok=True)
    return str(p)


def show_measurement_window(file_path: str, mask, poni, parent=None):
    """Placeholder UI function for showing a measurement window.
    No-op for now to satisfy imports.
    """
    return None


class CaptureWorker(QObject):
    """Minimal placeholder worker that simulates capture.
    Emits finished(success, result_files) when run().
    """

    finished = pyqtSignal(bool, dict)

    def __init__(self, detector_controller, integration_time: float, txt_filename_base: str):
        super().__init__()
        self.detector_controller = detector_controller
        self.integration_time = integration_time
        self.txt_filename_base = txt_filename_base

    def run(self):
        # Placeholder: immediately emit failure to indicate no real capture logic
        self.finished.emit(False, {})

