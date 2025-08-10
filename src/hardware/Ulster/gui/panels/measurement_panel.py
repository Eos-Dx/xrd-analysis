"""Measurement panel combining zone measurements and processing logic."""

import hashlib
import json
import time
from copy import copy
from pathlib import Path
from typing import Optional

from PyQt5.QtCore import QThread, QTimer
from PyQt5.QtGui import QColor
from PyQt5.QtWidgets import QMessageBox, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QSpinBox, QProgressBar

from core.measurement import CaptureWorker, MeasurementWorker
from core.geometry import sort_points_by_coordinates, convert_pixel_to_mm
from gui.widgets.measurement_widgets import MeasurementHistoryWidget
from utils.logging import get_module_logger

logger = get_module_logger(__name__)


class MeasurementPanel(QWidget):
    """Panel for managing zone measurements and processing."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.main_window = parent
        self.setup_ui()
        self.init_state()
        
    def setup_ui(self):
        """Setup the measurement panel UI."""
        layout = QVBoxLayout(self)
        
        # Control buttons
        button_layout = QHBoxLayout()
        self.start_btn = QPushButton("Start Measurements")
        self.pause_btn = QPushButton("Pause")  
        self.stop_btn = QPushButton("Stop")
        
        self.start_btn.clicked.connect(self.start_measurements)
        self.pause_btn.clicked.connect(self.pause_measurements)
        self.stop_btn.clicked.connect(self.stop_measurements)
        
        button_layout.addWidget(self.start_btn)
        button_layout.addWidget(self.pause_btn)
        button_layout.addWidget(self.stop_btn)
        layout.addLayout(button_layout)
        
        # Integration time
        time_layout = QHBoxLayout()
        time_layout.addWidget(QLabel("Integration time (s):"))
        self.integrationSpinBox = QSpinBox()
        self.integrationSpinBox.setMinimum(1)
        self.integrationSpinBox.setMaximum(300)
        self.integrationSpinBox.setValue(5)
        time_layout.addWidget(self.integrationSpinBox)
        layout.addLayout(time_layout)
        
        # Progress
        self.progressBar = QProgressBar()
        layout.addWidget(self.progressBar)
        
        self.timeRemainingLabel = QLabel("Ready")
        layout.addWidget(self.timeRemainingLabel)
        
    def init_state(self):
        """Initialize measurement state."""
        self.stopped = False
        self.paused = False
        self.current_measurement_sorted_index = 0
        self.total_points = 0
        self.measurement_widgets = {}
        
    def start_measurements(self):
        """Start the measurement process."""
        try:
            self.validate_and_prepare_measurements()
            self.sort_and_schedule_points()
            self.begin_measurement_sequence()
            
        except Exception as e:
            logger.error("Failed to start measurements", error=str(e))
            QMessageBox.critical(self, "Error", f"Failed to start measurements: {e}")
            
    def validate_and_prepare_measurements(self):
        """Validate measurement setup and prepare state."""
        # Get main window reference  
        main_window = self.main_window
        
        # Check measurement folder exists
        measurement_folder = Path(main_window.folderLineEdit.text().strip())
        if not measurement_folder.exists():
            raise ValueError("Measurement folder does not exist")
            
        # Check we have points to measure
        if main_window.pointsTable.rowCount() == 0:
            raise ValueError("No measurement points available")
            
        # Save current state
        main_window.manual_save_state()
        
        # Store references
        self.measurement_folder = measurement_folder
        self.state_path_measurements = (
            measurement_folder / f"{main_window.fileNameLineEdit.text()}_state.json"
        )
        
    def sort_and_schedule_points(self):
        """Sort points for efficient measurement order."""
        main_window = self.main_window
        
        # Collect all points
        generated_points = main_window.image_view.points_dict["generated"]["points"]
        user_points = main_window.image_view.points_dict["user"]["points"]
        
        all_points = []
        
        # Convert to measurement coordinates
        for i, item in enumerate(generated_points):
            center = item.sceneBoundingRect().center()
            x_mm, y_mm = convert_pixel_to_mm(
                (center.x(), center.y()),
                (main_window.real_x_pos_mm.value(), main_window.real_y_pos_mm.value()),
                main_window.include_center,
                main_window.pixel_to_mm_ratio
            )
            all_points.append((i, x_mm, y_mm))
            
        offset = len(generated_points)
        for j, item in enumerate(user_points):
            center = item.sceneBoundingRect().center()
            x_mm, y_mm = convert_pixel_to_mm(
                (center.x(), center.y()),
                (main_window.real_x_pos_mm.value(), main_window.real_y_pos_mm.value()),
                main_window.include_center,
                main_window.pixel_to_mm_ratio
            )
            all_points.append((offset + j, x_mm, y_mm))
            
        # Sort by coordinates for efficient measurement
        all_points_sorted = sorted(all_points, key=lambda tup: (tup[1], tup[2]))
        self.sorted_indices = [tup[0] for tup in all_points_sorted]
        self.total_points = len(self.sorted_indices)
        
        logger.info("Sorted measurement points", total_points=self.total_points)
        
    def begin_measurement_sequence(self):
        """Begin the actual measurement sequence."""
        self.start_btn.setEnabled(False)
        self.pause_btn.setEnabled(True)
        self.stop_btn.setEnabled(True)
        self.stopped = False
        self.paused = False
        self.current_measurement_sorted_index = 0
        
        self.progressBar.setMaximum(self.total_points)
        self.progressBar.setValue(0)
        
        integration_time = self.integrationSpinBox.value()
        initial_estimate = self.total_points * integration_time
        self.measurementStartTime = time.time()
        
        self.timeRemainingLabel.setText(f"Estimated time: {initial_estimate:.0f} sec")
        
        logger.info("Starting measurement sequence", 
                   total_points=self.total_points, 
                   integration_time=integration_time)
        
        # Start first measurement
        self.measure_next_point()
        
    def measure_next_point(self):
        """Measure the next point in the sequence."""
        if self.stopped or self.paused:
            return
            
        if self.current_measurement_sorted_index >= self.total_points:
            self.finish_measurements()
            return
            
        # Implementation continues with stage movement and capture...
        logger.debug("Measuring point", 
                    index=self.current_measurement_sorted_index,
                    total=self.total_points)
    
    def finish_measurements(self):
        """Complete the measurement sequence."""
        logger.info("All measurements completed")
        self.start_btn.setEnabled(True)
        self.pause_btn.setEnabled(False) 
        self.stop_btn.setEnabled(False)
        self.timeRemainingLabel.setText("Measurements complete")
        
    def pause_measurements(self):
        """Toggle pause/resume measurements."""
        if not self.paused:
            self.paused = True
            self.pause_btn.setText("Resume")
            logger.info("Measurements paused")
        else:
            self.paused = False
            self.pause_btn.setText("Pause")  
            logger.info("Measurements resumed")
            self.measure_next_point()
            
    def stop_measurements(self):
        """Stop measurements and reset."""
        self.stopped = True
        self.paused = False
        self.current_measurement_sorted_index = 0
        self.progressBar.setValue(0)
        self.timeRemainingLabel.setText("Measurements stopped")
        
        self.start_btn.setEnabled(True)
        self.pause_btn.setText("Pause")
        self.pause_btn.setEnabled(False)
        self.stop_btn.setEnabled(False)
        
        logger.info("Measurements stopped and reset")
