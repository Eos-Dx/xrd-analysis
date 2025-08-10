from PyQt5.QtCore import QObject, pyqtSignal


class MeasurementWorker(QObject):
    """
    Minimal placeholder processing worker used after capture completes.
    """

    add_aux_item = pyqtSignal(str, str)  # alias, npy_path

    def __init__(self, filenames):
        super().__init__()
        self.filenames = filenames or {}

    def run(self):
        # No-op: in real implementation this would process result files
        pass

