# zone_measurements/logic/utils.py

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPen


class ZoneMeasurementsUtilsMixin:
    def _add_beam_line(self, x1, y1, x2, y2, pen=None):
        """
        Adds a beam cross line to the scene at (x1, y1)-(x2, y2) using the provided pen.
        Returns the QGraphicsLineItem instance.
        """
        from PyQt5.QtWidgets import QGraphicsLineItem

        if pen is None:
            pen = QPen(Qt.black, 5)

        line = QGraphicsLineItem(x1, y1, x2, y2)
        line.setPen(pen)
        self.image_view.scene.addItem(line)
        return line

    def mm_to_pixels(self, x_mm: float, y_mm: float):
        """
        Converts stage X/Y coordinates in mm to image pixel coordinates.
        Uses instance variables for real position, scaling, and center.
        """
        x = (
            self.real_x_pos_mm.value() - x_mm
        ) * self.pixel_to_mm_ratio + self.include_center[0]
        y = (
            self.real_y_pos_mm.value() - y_mm
        ) * self.pixel_to_mm_ratio + self.include_center[1]
        return x, y

    # Add other reusable utilities as staticmethods or instance methods here if needed.
