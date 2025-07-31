# zone_measurements/logic/process_mixin.py

import time
from copy import copy

from PyQt5.QtCore import QThread, QTimer
from PyQt5.QtGui import QColor

from hardware.Ulster.gui.technical.capture import (
    CaptureWorker,
    validate_folder,
)
from hardware.Ulster.gui.technical.measurement_worker import MeasurementWorker
from hardware.Ulster.gui.technical.widgets import MeasurementHistoryWidget


class ZoneMeasurementsProcessMixin:
    def start_measurements(self):
        """
        Starts the measurements for all sorted points.
        Prepares measurement folder, state, sorts points, starts progress bar, etc.
        """
        # Folder validation and state saving
        self.measurement_folder = validate_folder(
            self.folderLineEdit.text().strip()
        )
        self.state_path_measurements = (
            self.measurement_folder
            / f"{self.fileNameLineEdit.text()}_state.json"
        )
        self.manual_save_state()

        self.state_measurements = copy(self.state)
        try:
            from hardware.Ulster.hardware.auxiliary import (
                encode_image_to_base64,
            )

            self.state_measurements["image_base64"] = encode_image_to_base64(
                self.image_view.current_image_path
            )
            with open(self.state_path_measurements, "w") as f:
                import json

                json.dump(self.state_measurements, f, indent=4)
        except Exception as e:
            print(e)

        if self.pointsTable.rowCount() == 0:
            print("No points available for measurement.")
            return

        self.start_btn.setEnabled(False)
        self.pause_btn.setEnabled(True)
        self.stop_btn.setEnabled(True)
        self.stopped = False
        self.paused = False

        # Consolidate and sort measurement points
        generated_points = self.image_view.points_dict["generated"]["points"]
        user_points = self.image_view.points_dict["user"]["points"]
        all_points = []
        for i, item in enumerate(generated_points):
            center = item.sceneBoundingRect().center()
            x_mm = (
                self.real_x_pos_mm.value()
                - (center.x() - self.include_center[0])
                / self.pixel_to_mm_ratio
            )
            y_mm = (
                self.real_y_pos_mm.value()
                - (center.y() - self.include_center[1])
                / self.pixel_to_mm_ratio
            )
            all_points.append((i, x_mm, y_mm))
        offset = len(generated_points)
        for j, item in enumerate(user_points):
            center = item.sceneBoundingRect().center()
            x_mm = (
                self.real_x_pos_mm.value()
                - (center.x() - self.include_center[0])
                / self.pixel_to_mm_ratio
            )
            y_mm = (
                self.real_y_pos_mm.value()
                - (center.y() - self.include_center[1])
                / self.pixel_to_mm_ratio
            )
            all_points.append((offset + j, x_mm, y_mm))
        all_points_sorted = sorted(
            all_points, key=lambda tup: (tup[1], tup[2])
        )
        self.sorted_indices = [tup[0] for tup in all_points_sorted]
        self.total_points = len(self.sorted_indices)
        self.current_measurement_sorted_index = 0

        self.progressBar.setMaximum(self.total_points)
        self.progressBar.setValue(0)
        self.integration_time = self.integrationSpinBox.value()
        self.initial_estimate = self.total_points * self.integration_time
        self.measurementStartTime = time.time()
        self.timeRemainingLabel.setText(
            f"Estimated time: {self.initial_estimate:.0f} sec"
        )
        print("Starting measurements in sorted order...")
        self.measure_next_point()

    def measure_next_point(self):
        """
        Moves to the next point and triggers capture on both detectors.
        Advances through sorted_indices and updates progress.
        """
        if self.stopped:
            print("Measurement stopped.")
            return
        if self.paused:
            print("Measurement is paused. Waiting for resume.")
            return
        if self.current_measurement_sorted_index >= self.total_points:
            print("All points measured.")
            self.start_btn.setEnabled(True)
            self.pause_btn.setEnabled(False)
            self.stop_btn.setEnabled(False)
            return

        index = self.sorted_indices[self.current_measurement_sorted_index]
        gp = self.image_view.points_dict["generated"]["points"]
        up = self.image_view.points_dict["user"]["points"]
        if index < len(gp):
            self._point_item = gp[index]
            self._zone_item = self.image_view.points_dict["generated"][
                "zones"
            ][index]
        else:
            user_index = index - len(gp)
            self._point_item = up[user_index]
            self._zone_item = self.image_view.points_dict["user"]["zones"][
                user_index
            ]

        self.update_xy_pos()
        center = self._point_item.sceneBoundingRect().center()
        self._x_mm = (
            self.real_x_pos_mm.value()
            - (center.x() - self.include_center[0]) / self.pixel_to_mm_ratio
        )
        self._y_mm = (
            self.real_y_pos_mm.value()
            - (center.y() - self.include_center[1]) / self.pixel_to_mm_ratio
        )

        # Move the stage using the controller
        new_x, new_y = self.stage_controller.move_stage(
            self._x_mm, self._y_mm, move_timeout=10
        )

        # Build a common filename base (without extension or detector label)
        import os
        import time

        self._timestamp = time.strftime("%Y%m%d_%H%M%S")
        self._base_name = self.fileNameLineEdit.text().strip()
        txt_filename_base = os.path.join(
            self.measurement_folder,
            f"{self._base_name}_{self._x_mm:.2f}_{self._y_mm:.2f}_{self._timestamp}",
        )

        # Launch the dual-capture worker in its own thread
        self.capture_worker = CaptureWorker(
            detector_controller=self.detector_controller,
            integration_time=self.integration_time,
            txt_filename_base=txt_filename_base,
        )
        self.capture_thread = QThread()
        self.capture_worker.moveToThread(self.capture_thread)
        self.capture_thread.started.connect(self.capture_worker.run)
        self.capture_worker.finished.connect(self.on_capture_finished)
        self.capture_worker.finished.connect(self.capture_thread.quit)
        self.capture_worker.finished.connect(self.capture_worker.deleteLater)
        self.capture_thread.finished.connect(self.capture_thread.deleteLater)
        self.capture_thread.start()

    def on_capture_finished(self, success: bool, result_files: dict):
        """
        Callback after detector(s) finish capturing.
        Handles errors, triggers post-processing, colors UI.
        """
        if not success:
            print("[Measurement] capture failed.")
            return
        print(f"[Measurement] capture successful: {result_files}")

        current_row = self.sorted_indices[
            self.current_measurement_sorted_index
        ]
        self.spawn_measurement_thread(current_row, result_files)

        # Visual feedback
        green_brush = QColor(0, 255, 0)
        self._point_item.setBrush(green_brush)
        try:
            if self._zone_item:
                green_zone = QColor(0, 255, 0)
                green_zone.setAlphaF(0.2)
                self._zone_item.setBrush(green_zone)
        except Exception as e:
            print(e)
        QTimer.singleShot(1000, self.measurement_finished)

    def spawn_measurement_thread(self, row, file_map):
        """
        Spawns a MeasurementWorker in a new thread for post-processing measurement files.
        Connects signals for result handling and thread cleanup.
        """
        thread = QThread(self)
        worker = MeasurementWorker(
            row=row,
            filenames=file_map,
            masks=self.masks,
            ponis=self.ponis,
            parent=self,
            hf_cutoff_fraction=0.2,
            columns_to_remove=30,
        )
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.measurement_ready.connect(self.add_measurement_to_table)
        worker.measurement_ready.connect(thread.quit)
        worker.measurement_ready.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)
        if not hasattr(self, "_measurement_threads"):
            self._measurement_threads = []
        self._measurement_threads.append((thread, worker))
        thread.start()

    def measurement_finished(self):
        """
        Called after one measurement completes.
        Advances progress, updates time estimates, and triggers next point if not done.
        """
        if self.stopped:
            print("Measurement stopped.")
            return

        self.current_measurement_sorted_index += 1
        self.progressBar.setValue(self.current_measurement_sorted_index)
        elapsed = time.time() - self.measurementStartTime
        if self.current_measurement_sorted_index > 0:
            avg_time = elapsed / self.current_measurement_sorted_index
            remaining = avg_time * (
                self.total_points - self.current_measurement_sorted_index
            )
            percent_complete = (
                self.current_measurement_sorted_index / self.total_points
            ) * 100
            self.timeRemainingLabel.setText(
                f"{percent_complete:.0f}% done, {remaining:.0f} sec remaining"
            )

        if (
            self.current_measurement_sorted_index < self.total_points
            and not self.paused
            and not self.stopped
        ):
            self.measure_next_point()
        else:
            if self.current_measurement_sorted_index >= self.total_points:
                print("All points measured.")
                self.pause_btn.setEnabled(False)
                self.stop_btn.setEnabled(False)
                self.start_btn.setEnabled(True)

    def add_measurement_to_table(self, row, results, timestamp=None):
        """
        Adds the measurement results to the points table at the specified row,
        creates/updates the associated MeasurementHistoryWidget.
        """
        widget = self.pointsTable.cellWidget(row, 5)
        if not isinstance(widget, MeasurementHistoryWidget):
            widget = MeasurementHistoryWidget(
                masks=self.masks, ponis=self.ponis, parent=self
            )
            self.pointsTable.setCellWidget(row, 5, widget)
            self.measurement_widgets[row] = widget
        widget.add_measurement(
            results, timestamp or getattr(self, "_timestamp", "")
        )

    def pause_measurements(self):
        """
        Toggles between pause and resume of the measurement sequence.
        """
        if not hasattr(self, "paused"):
            self.paused = False
        if not self.paused:
            self.paused = True
            self.pause_btn.setText("Resume")
            print("Measurements paused.")
        else:
            self.paused = False
            self.pause_btn.setText("Pause")
            print("Measurements resumed.")
            self.measure_next_point()

    def stop_measurements(self):
        """
        Stops the measurement process, resets all progress/UI states.
        """
        self.stopped = True
        self.paused = False
        self.current_measurement_sorted_index = 0
        self.progressBar.setValue(0)
        self.timeRemainingLabel.setText("Measurement stopped.")
        self.start_btn.setEnabled(True)
        self.pause_btn.setText("Pause")
        self.pause_btn.setEnabled(False)
        self.stop_btn.setEnabled(False)
        print("Measurements stopped and reset.")
