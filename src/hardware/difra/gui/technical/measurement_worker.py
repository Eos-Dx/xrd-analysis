# measurement_worker.py
import time
from pathlib import Path

from PyQt5.QtCore import QObject, pyqtSignal, pyqtSlot  # <-- add pyqtSlot
from PyQt5.QtWidgets import QListWidgetItem

from hardware.difra.gui.technical.capture import (
    compute_hf_score_from_cake,
)
from hardware.difra.utils.logger import get_module_logger

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
        filenames,  # dict: {alias: filename} - already converted to container format by detector
        masks=None,  # dict: {alias: mask}
        ponis=None,  # dict: {alias: poni}
        row=None,  # int: row in table, for zone measurements
        parent=None,
        hf_cutoff_fraction=0.2,
        columns_to_remove=30,
        frames: int = 1,  # Deprecated - conversion now handled by detector
        average_frames: bool = False,  # Deprecated - conversion now handled by detector
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
        self.frames = int(frames) if frames is not None else 1
        self.average_frames = bool(average_frames)

    @pyqtSlot()
    def run(self):
        """Process already-converted measurement files.
        
        Files are already in container format (.npy for v0.2) - conversion
        is handled by the detector's convert_to_container_format() method.
        This worker just computes quality metrics and emits results.
        """
        results = {}
        for alias, container_file in self.filenames.items():
            logger.debug(
                f"Processing detector measurement", detector=alias, file=container_file
            )
            
            # Files are already converted by detector - just use them
            npy_path = str(container_file)
            
            # Emit for GUI table
            self.add_aux_item.emit(alias, npy_path)
            
            # Compute quality metrics
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
