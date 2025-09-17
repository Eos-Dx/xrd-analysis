"""UI builder for zone points functionality."""

import math
from typing import List, Tuple

from PyQt5.QtCore import QPointF
from PyQt5.QtWidgets import (
    QDoubleSpinBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSpinBox,
    QTableWidget,
)

from hardware.Ulster.gui.main_window_ext.points.zone_geometry import (
    sample_points_in_circle,
    sample_points_in_rect,
)

from .zone_points_constants import ZonePointsConstants


class ZonePointsUIBuilder:
    """Responsible for building the zone points UI components."""

    @staticmethod
    def create_controls_layout(parent) -> QHBoxLayout:
        """Create the input controls layout."""
        layout = QHBoxLayout()
        try:
            layout.setContentsMargins(0, 0, 0, 0)
            layout.setSpacing(6)
        except Exception:
            pass

        # Point count controls
        layout.addWidget(QLabel("N points"))
        parent.pointCountSpinBox = QSpinBox()
        try:
            parent.pointCountSpinBox.setFixedHeight(22)
        except Exception:
            pass
        parent.pointCountSpinBox.setMinimum(ZonePointsConstants.MIN_POINTS)
        parent.pointCountSpinBox.setMaximum(ZonePointsConstants.MAX_POINTS)
        parent.pointCountSpinBox.setValue(ZonePointsConstants.DEFAULT_POINTS)
        layout.addWidget(parent.pointCountSpinBox)

        # Shrink controls
        layout.addWidget(QLabel("% offset"))
        parent.shrinkSpinBox = QSpinBox()
        try:
            parent.shrinkSpinBox.setFixedHeight(22)
        except Exception:
            pass
        parent.shrinkSpinBox.setMinimum(0)
        parent.shrinkSpinBox.setMaximum(100)
        parent.shrinkSpinBox.setValue(ZonePointsConstants.DEFAULT_SHRINK_PERCENT)
        layout.addWidget(parent.shrinkSpinBox)

        return layout

    @staticmethod
    def create_coordinate_controls(parent) -> QHBoxLayout:
        """Create coordinate input controls."""
        layout = QHBoxLayout()
        try:
            layout.setContentsMargins(0, 0, 0, 0)
            layout.setSpacing(6)
        except Exception:
            pass

        real_x, real_y = ZonePointsUIBuilder._get_real_zero_coordinates(parent)

        # X coordinate
        parent.realXLabel = QLabel("X_pos, mm")
        layout.addWidget(parent.realXLabel)
        parent.real_x_pos_mm = QDoubleSpinBox()
        parent.real_x_pos_mm.setDecimals(ZonePointsConstants.COORDINATE_DECIMALS)
        parent.real_x_pos_mm.setRange(*ZonePointsConstants.COORDINATE_RANGE)
        parent.real_x_pos_mm.setValue(real_x)
        try:
            parent.real_x_pos_mm.setFixedHeight(22)
        except Exception:
            pass
        layout.addWidget(parent.real_x_pos_mm)

        # Y coordinate
        parent.realYLabel = QLabel("Y_pos, mm")
        layout.addWidget(parent.realYLabel)
        parent.real_y_pos_mm = QDoubleSpinBox()
        parent.real_y_pos_mm.setDecimals(ZonePointsConstants.COORDINATE_DECIMALS)
        parent.real_y_pos_mm.setRange(*ZonePointsConstants.COORDINATE_RANGE)
        parent.real_y_pos_mm.setValue(real_y)
        try:
            parent.real_y_pos_mm.setFixedHeight(22)
        except Exception:
            pass
        layout.addWidget(parent.real_y_pos_mm)

        # Conversion label
        parent.conversionLabel = QLabel("Conversion: 1.00 px/mm")
        layout.addWidget(parent.conversionLabel)

        return layout

    @staticmethod
    def create_action_buttons(parent) -> QHBoxLayout:
        """Create action buttons."""
        layout = QHBoxLayout()
        try:
            layout.setContentsMargins(0, 0, 0, 0)
            layout.setSpacing(6)
        except Exception:
            pass

        parent.generatePointsBtn = QPushButton("Generate Points")
        layout.addWidget(parent.generatePointsBtn)

        parent.updateCoordinatesBtn = QPushButton("Update Coordinates")
        layout.addWidget(parent.updateCoordinatesBtn)

        # Add current position display near Update Coordinates button
        if not hasattr(parent, "zoneCurrentPositionLabel"):
            parent.zoneCurrentPositionLabel = QLabel("Current XY: (Not initialized)")
            parent.zoneCurrentPositionLabel.setStyleSheet(
                "color: #666; font-size: 9px; margin: 1px;"
            )
        layout.addWidget(parent.zoneCurrentPositionLabel)

        return layout

    @staticmethod
    def create_points_table(parent) -> QTableWidget:
        """Create the points table."""
        table = QTableWidget(0, len(ZonePointsConstants.TABLE_COLUMNS))
        table.setHorizontalHeaderLabels(ZonePointsConstants.TABLE_COLUMNS)
        return table

    @staticmethod
    def _get_real_zero_coordinates(parent) -> Tuple[float, float]:
        """Get real zero coordinates from config."""
        real_x, real_y = (
            ZonePointsConstants.DEFAULT_REAL_X,
            ZonePointsConstants.DEFAULT_REAL_Y,
        )

        try:
            active_stage_ids = parent.config.get("active_translation_stages", [])
            translation_stages = parent.config.get("translation_stages", [])
            active_id = active_stage_ids[0] if active_stage_ids else None

            for stage in translation_stages:
                if stage.get("id") == active_id:
                    real_zero = stage.get("real_zero", {})
                    if real_zero:
                        real_x = real_zero.get("x_mm", real_x)
                        real_y = real_zero.get("y_mm", real_y)
                    break
        except Exception as e:
            print(f"Error fetching real_zero for active stage: {e}")

        return real_x, real_y


class ZonePointsGeometry:
    """Handles geometric calculations for zone points."""

    @staticmethod
    def get_shape_bounds_and_candidates(
        include_shape, shrink_factor: float
    ) -> Tuple[List[Tuple[float, float]], float, Tuple[float, float, float, float]]:
        """Get candidate points and bounds for a shape."""
        if hasattr(include_shape, "center") and hasattr(include_shape, "radius"):
            return ZonePointsGeometry._get_circle_candidates(
                include_shape, shrink_factor
            )
        else:
            return ZonePointsGeometry._get_rect_candidates(include_shape, shrink_factor)

    @staticmethod
    def _get_circle_candidates(
        include_shape, shrink_factor: float
    ) -> Tuple[List[Tuple[float, float]], float, Tuple[float, float, float, float]]:
        """Get candidates for circular shape."""
        center = include_shape.center
        radius = include_shape.radius * shrink_factor
        candidates = sample_points_in_circle(
            center, radius, ZonePointsConstants.MAX_CANDIDATES
        )
        area = math.pi * (radius**2)
        bounds = (
            center[0] - radius,
            center[1] - radius,
            center[0] + radius,
            center[1] + radius,
        )
        return candidates, area, bounds

    @staticmethod
    def _get_rect_candidates(
        include_shape, shrink_factor: float
    ) -> Tuple[List[Tuple[float, float]], float, Tuple[float, float, float, float]]:
        """Get candidates for rectangular shape."""
        rect = include_shape.boundingRect()
        x_min, y_min = rect.x(), rect.y()
        x_max, y_max = x_min + rect.width(), y_min + rect.height()
        candidates = sample_points_in_rect(
            x_min, y_min, x_max, y_max, ZonePointsConstants.MAX_CANDIDATES
        )
        area = (x_max - x_min) * (y_max - y_min)
        bounds = (x_min, y_min, x_max, y_max)
        return candidates, area, bounds

    @staticmethod
    def filter_candidates_by_shapes(
        candidates: List[Tuple[float, float]], include_shape, exclude_shapes: List
    ) -> List[Tuple[float, float]]:
        """Filter candidate points by inclusion/exclusion shapes."""
        filtered = []
        for pt in candidates:
            ptf = QPointF(*pt)
            if not include_shape.contains(include_shape.mapFromScene(ptf)):
                continue
            if any(ex.contains(ex.mapFromScene(ptf)) for ex in exclude_shapes):
                continue
            filtered.append(pt)
        return filtered
