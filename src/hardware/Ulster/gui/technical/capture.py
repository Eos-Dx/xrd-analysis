import os
from PyQt5.QtCore import  QThread, pyqtSignal

class CaptureWorker(QThread):
    # emit (success: bool, txt_filename: str)
    finished = pyqtSignal(bool, str)

    def __init__(self, detector_controller, integration_time, txt_filename, parent=None):
        super().__init__(parent)
        self.detector_controller = detector_controller
        self.integration_time = integration_time
        self.txt_filename = txt_filename

    def run(self):
        # this executes in the worker thread
        success = self.detector_controller.capture_point(
            1,
            self.integration_time,
            self.txt_filename
        )
        # emit back to the GUI thread
        self.finished.emit(success, self.txt_filename)


def validate_folder(path: str) -> str:
    if not path:
        path = os.getcwd()
    try:
        os.makedirs(path, exist_ok=True)
    except Exception:
        path = os.getcwd()
    if not os.access(path, os.W_OK):
        path = os.getcwd()
    return path
