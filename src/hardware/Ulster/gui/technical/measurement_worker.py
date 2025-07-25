# measurement_worker.py
from PyQt5.QtCore import QObject, pyqtSignal


class MeasurementWorker(QObject):
    measurement_ready = pyqtSignal(int, str, float, int)

    def __init__(
        self,
        row,
        measurement_filename,
        mask,
        poni,
        parent,
        hf_cutoff_fraction=0.2,
        columns_to_remove=30,
    ):
        super().__init__()
        self.row = row
        self.measurement_filename = measurement_filename
        self.mask = mask
        self.poni = poni
        self.parent = parent
        self.hf_cutoff_fraction = hf_cutoff_fraction
        self.columns_to_remove = columns_to_remove
        self.goodness = 0

    def run(self):
        from hardware.Ulster.gui.technical.capture import (
            compute_hf_score_from_cake,
        )

        self.goodness = compute_hf_score_from_cake(
            self.measurement_filename,
            self.poni,
            self.mask,
            hf_cutoff_fraction=self.hf_cutoff_fraction,
            skip_bins=self.columns_to_remove,
        )
        self.measurement_ready.emit(
            self.row,
            self.measurement_filename,
            self.goodness,
            self.columns_to_remove,
        )
