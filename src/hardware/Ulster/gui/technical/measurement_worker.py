# measurement_worker.py
from PyQt5.QtCore import QObject, pyqtSignal
from time import time

from PyQt5.QtCore import QObject, pyqtSignal
import time

class MeasurementWorker(QObject):
    # Emits: row (int), results (dict), timestamp (str)
    measurement_ready = pyqtSignal(int, dict, str)

    def __init__(
        self,
        row,
        filenames,  # dict: {alias: filename}
        masks,      # dict: {alias: mask}
        ponis,      # dict: {alias: poni}
        parent,
        hf_cutoff_fraction=0.2,
        columns_to_remove=30,
    ):
        super().__init__()
        self.row = row
        self.filenames = filenames
        self.masks = masks
        self.ponis = ponis
        self.parent = parent
        self.hf_cutoff_fraction = hf_cutoff_fraction
        self.columns_to_remove = columns_to_remove

    def run(self):
        from hardware.Ulster.gui.technical.capture import compute_hf_score_from_cake
        results = {}
        for alias, filename in self.filenames.items():
            mask = self.masks.get(alias)
            poni = self.ponis.get(alias)
            try:
                goodness = compute_hf_score_from_cake(
                    filename,
                    poni,
                    mask,
                    hf_cutoff_fraction=self.hf_cutoff_fraction,
                    skip_bins=self.columns_to_remove,
                )
            except Exception as e:
                print(f"[{alias}] Error in compute_hf_score_from_cake: {e}")
                goodness = float("nan")
            results[alias] = {
                "filename": filename,
                "goodness": goodness
            }
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        self.measurement_ready.emit(self.row, results, timestamp)
