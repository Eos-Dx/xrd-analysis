# measurement_worker.py
import time
from pathlib import Path

from PyQt5.QtCore import QObject, pyqtSignal, pyqtSlot  # <-- add pyqtSlot
from PyQt5.QtWidgets import QListWidgetItem
from utils.logging import get_module_logger

from .processor import compute_hf_score_from_cake, move_and_convert_measurement_file

logger = get_module_logger(__name__)


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
        logger.debug(
            "MeasurementWorker constructed", row=row, detectors=list(filenames.keys())
        )
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
            logger.debug(
                f"Processing detector measurement", detector=alias, file=txt_file
            )
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
                logger.debug(
                    f"Computed goodness score", detector=alias, goodness=goodness
                )
            except Exception as e:
                logger.error(
                    f"Error computing goodness score", detector=alias, error=str(e)
                )
                goodness = float("nan")
            results[alias] = {"filename": str(npy_path), "goodness": goodness}
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        if self.row is not None:
            logger.info(
                f"Measurement completed for zone row",
                row=self.row,
                detectors=list(results.keys()),
            )
            self.measurement_ready.emit(self.row, results, timestamp)
        logger.debug("MeasurementWorker finished")
        self.finished.emit()
