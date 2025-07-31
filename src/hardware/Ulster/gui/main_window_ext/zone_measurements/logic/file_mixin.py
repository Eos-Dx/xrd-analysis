# zone_measurements/logic/file_mixin.py

import json
import os
from pathlib import Path

import numpy as np
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QFileDialog, QListWidgetItem


class ZoneMeasurementsFileMixin:
    def browse_folder(self):
        """
        Open a dialog to choose the save folder and set it in the UI.
        """
        folder = QFileDialog.getExistingDirectory(self, "Select Folder")
        if folder:
            self.folderLineEdit.setText(folder)


    def process_measurement_result(self, success, result_files, typ):
        """
        Handles measurement files, organizes by alias, converts .txt to .npy,
        and updates the UI list widget. Used for auxiliary or technical measurements.
        """
        if not success:
            print(f"[{typ}] capture failed.")
            if hasattr(self, "_aux_timer"):
                self._aux_timer.stop()
            if hasattr(self, "_aux_status"):
                self._aux_status.setText("")
            return {}
        else:
            print(f"[{typ}] capture successful: {result_files}")
            if hasattr(self, "_aux_timer"):
                self._aux_timer.stop()
            if hasattr(self, "_aux_status"):
                self._aux_status.setText("Done")

        file_map = {}
        for alias, txt_file in result_files.items():
            txt_path = Path(txt_file)
            det_folder = txt_path.parent / alias
            det_folder.mkdir(parents=True, exist_ok=True)
            new_txt_file = det_folder / txt_path.name
            try:
                txt_path.replace(new_txt_file)
            except Exception as e:
                print(f"[ERROR] Moving file {txt_path} → {new_txt_file}: {e}")
                new_txt_file = txt_path
            try:
                data = np.loadtxt(new_txt_file)
                npy = new_txt_file.with_suffix(".npy")
                np.save(npy, data)
            except Exception as e:
                print(f"Conversion error for {alias}: {e}")
                npy = new_txt_file
            file_map[alias] = str(npy)
            if hasattr(self, "auxList"):
                item = QListWidgetItem(f"{alias}: {Path(npy).name}")
                item.setData(Qt.UserRole, str(npy))
                self.auxList.addItem(item)
        return file_map

    def handle_add_count(self):
        """
        Appends the count value to the current file name in the UI.
        """
        current_filename = self.fileNameLineEdit.text()
        appended_value = "_" + str(self.addCountSpinBox.value())
        self.fileNameLineEdit.setText(current_filename + appended_value)

    def handle_add_distance(self):
        """
        Appends the distance value to the current file name in the UI.
        """
        current_filename = self.fileNameLineEdit.text()
        appended_value = "_" + self.add_distance_lineedit.text()
        self.fileNameLineEdit.setText(current_filename + appended_value)
