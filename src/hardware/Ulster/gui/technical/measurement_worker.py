# measurement_worker.py
import time
from pathlib import Path

from PyQt5.QtCore import QObject, pyqtSignal, pyqtSlot  # <-- add pyqtSlot
from PyQt5.QtWidgets import QListWidgetItem

from hardware.Ulster.gui.technical.capture import (
    compute_hf_score_from_cake,
    move_and_convert_measurement_file,
)


class MeasurementWorker(QObject):
    # For zone measurements: emits row (int), results (dict), timestamp (str)
    measurement_ready = pyqtSignal(int, dict, str)
    # For auxiliary measurements: emits when done
    finished = pyqtSignal()
    # Signal to update aux list in the main (GUI) thread
    add_aux_item = pyqtSignal(str, str)  # alias, npy_path

    def __init__(
        self,
        filenames,  # dict: {alias: filename}
        masks=None,  # dict: {alias: mask}
        ponis=None,  # dict: {alias: poni}
        row=None,  # int: row in table, for zone measurements
        parent=None,
        hf_cutoff_fraction=0.2,
        columns_to_remove=30,
    ):
        super().__init__(parent)
        print("MeasurementWorker constructed")
        self.row = row
        self.filenames = filenames
        self.masks = masks or {}
        self.ponis = ponis or {}
        self.hf_cutoff_fraction = hf_cutoff_fraction
        self.columns_to_remove = columns_to_remove

    @pyqtSlot()
    def run(self):

        results = {}
        for alias, txt_file in self.filenames.items():
            print(f"[MeasurementWorker] Processing {alias}: {txt_file}")
            src_path = Path(txt_file)
            alias_folder = src_path.parent / alias
            npy_path = move_and_convert_measurement_file(src_path, alias_folder)
            self.add_aux_item.emit(alias, str(npy_path))
            mask = self.masks.get(alias)
            poni = self.ponis.get(alias)
            try:
                goodness = compute_hf_score_from_cake(
                    npy_path,
                    poni,
                    mask,
                    hf_cutoff_fraction=self.hf_cutoff_fraction,
                    skip_bins=self.columns_to_remove,
                )
            except Exception as e:
                print(f"[{alias}] Error in compute_hf_score_from_cake: {e}")
                goodness = float("nan")
            results[alias] = {"filename": str(npy_path), "goodness": goodness}
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        if self.row is not None:
            print(f"[MeasurementWorker] Emitting measurement_ready for row {self.row}")
            self.measurement_ready.emit(self.row, results, timestamp)
        print("[MeasurementWorker] Emitting finished")
        self.finished.emit()
