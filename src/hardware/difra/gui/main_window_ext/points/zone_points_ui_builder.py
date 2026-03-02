"""UI builder for zone points functionality."""

import math
from typing import List, Optional, Tuple

from PyQt5.QtCore import QPointF, QSize, Qt
from PyQt5.QtWidgets import (
    QApplication,
    QDoubleSpinBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSpinBox,
    QStyle,
    QStyledItemDelegate,
    QStyleOptionViewItem,
    QTableWidget,
)

from hardware.difra.gui.main_window_ext.points.zone_geometry import (
    sample_points_in_circle,
    sample_points_in_ellipse,
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
        table.setItemDelegateForColumn(0, PointIdentityDelegate(table))
        try:
            table.verticalHeader().setDefaultSectionSize(42)
        except Exception:
            pass
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
        include_shape,
        shrink_factor: float,
        edge_clearance_px: float = 0.0,
    ) -> Tuple[List[Tuple[float, float]], float, Tuple[float, float, float, float]]:
        """Get candidate points and bounds for a shape."""
        ellipse_params = ZonePointsGeometry._extract_ellipse_params(include_shape)
        if ellipse_params is not None:
            center, radius_x, radius_y = ellipse_params
            return ZonePointsGeometry._get_ellipse_candidates(
                center,
                radius_x,
                radius_y,
                shrink_factor,
                edge_clearance_px=edge_clearance_px,
            )
        return ZonePointsGeometry._get_rect_candidates(
            include_shape,
            shrink_factor,
            edge_clearance_px=edge_clearance_px,
        )

    @staticmethod
    def _extract_ellipse_params(
        include_shape,
    ) -> Optional[Tuple[Tuple[float, float], float, float]]:
        """Return ``((center_x, center_y), radius_x, radius_y)`` for ellipse-like shapes."""
        # Preferred interface used by ResizableZoneItem.
        get_center = getattr(include_shape, "get_center", None)
        get_radius = getattr(include_shape, "get_radius", None)
        if callable(get_center) and callable(get_radius):
            try:
                center = get_center()
                radius = float(get_radius())
                if center is not None and len(center) >= 2 and radius > 0:
                    return (float(center[0]), float(center[1])), radius, radius
            except Exception:
                pass

        # Backward-compatible dynamic attributes.
        if hasattr(include_shape, "center") and hasattr(include_shape, "radius"):
            try:
                center = include_shape.center
                radius = float(include_shape.radius)
                if center is not None and len(center) >= 2 and radius > 0:
                    return (float(center[0]), float(center[1])), radius, radius
            except Exception:
                pass

        # Fallback for generic ellipse items (QGraphicsEllipseItem or subclasses).
        class_name = include_shape.__class__.__name__.lower()
        is_ellipse_like = ("ellipse" in class_name) or ("circle" in class_name)
        if is_ellipse_like:
            try:
                if hasattr(include_shape, "sceneBoundingRect"):
                    rect = include_shape.sceneBoundingRect()
                elif hasattr(include_shape, "rect") and callable(include_shape.rect):
                    rect = include_shape.rect()
                else:
                    return None

                w = float(rect.width())
                h = float(rect.height())
                if w > 0 and h > 0:
                    cx = float(rect.x()) + w / 2.0
                    cy = float(rect.y()) + h / 2.0
                    return (cx, cy), w / 2.0, h / 2.0
            except Exception:
                pass

        return None

    @staticmethod
    def _get_ellipse_candidates(
        center: Tuple[float, float],
        radius_x: float,
        radius_y: float,
        shrink_factor: float,
        edge_clearance_px: float = 0.0,
    ) -> Tuple[List[Tuple[float, float]], float, Tuple[float, float, float, float]]:
        """Get candidates for ellipse/circle shape."""
        shrink_factor = max(0.0, min(1.0, float(shrink_factor)))
        clearance = max(0.0, float(edge_clearance_px))
        radius_x = max(0.0, radius_x * shrink_factor - clearance)
        radius_y = max(0.0, radius_y * shrink_factor - clearance)
        candidates = sample_points_in_ellipse(
            center, radius_x, radius_y, ZonePointsConstants.MAX_CANDIDATES
        )
        area = math.pi * radius_x * radius_y
        bounds = (
            center[0] - radius_x,
            center[1] - radius_y,
            center[0] + radius_x,
            center[1] + radius_y,
        )
        return candidates, area, bounds

    @staticmethod
    def _get_circle_candidates(
        center: Tuple[float, float], radius: float, shrink_factor: float
    ) -> Tuple[List[Tuple[float, float]], float, Tuple[float, float, float, float]]:
        """Get candidates for circular shape."""
        candidates, area, bounds = ZonePointsGeometry._get_ellipse_candidates(
            center,
            radius,
            radius,
            shrink_factor,
            edge_clearance_px=0.0,
        )
        return candidates, area, bounds

    @staticmethod
    def _get_rect_candidates(
        include_shape,
        shrink_factor: float,
        edge_clearance_px: float = 0.0,
    ) -> Tuple[List[Tuple[float, float]], float, Tuple[float, float, float, float]]:
        """Get candidates for rectangular shape."""
        shrink_factor = max(0.0, min(1.0, float(shrink_factor)))
        clearance = max(0.0, float(edge_clearance_px))
        rect = include_shape.boundingRect()
        x_min, y_min = rect.x(), rect.y()
        x_max, y_max = x_min + rect.width(), y_min + rect.height()
        center_x = (x_min + x_max) / 2.0
        center_y = (y_min + y_max) / 2.0
        half_w = (rect.width() * shrink_factor) / 2.0
        half_h = (rect.height() * shrink_factor) / 2.0
        x_min, x_max = center_x - half_w, center_x + half_w
        y_min, y_max = center_y - half_h, center_y + half_h
        x_min += clearance
        x_max -= clearance
        y_min += clearance
        y_max -= clearance
        if x_min > x_max:
            x_min = x_max = center_x
        if y_min > y_max:
            y_min = y_max = center_y
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


class PointIdentityDelegate(QStyledItemDelegate):
    """Render point display number and UID in a compact two-line style."""

    def paint(self, painter, option, index):
        if index.column() != 0:
            super().paint(painter, option, index)
            return

        style_option = QStyleOptionViewItem(option)
        self.initStyleOption(style_option, index)
        style_option.text = ""

        style = (
            style_option.widget.style()
            if style_option.widget is not None
            else QApplication.style()
        )
        style.drawControl(
            QStyle.CE_ItemViewItem, style_option, painter, style_option.widget
        )

        display_text = str(index.data(Qt.DisplayRole) or "").strip()
        uid_text = str(index.data(Qt.UserRole + 1) or "").strip()
        if not uid_text:
            uid_text = "uid: -"

        painter.save()
        content_rect = style.subElementRect(
            QStyle.SE_ItemViewItemText, style_option, style_option.widget
        )

        number_font = style_option.font
        number_font.setBold(True)
        number_font.setPointSize(max(8, number_font.pointSize()))
        painter.setFont(number_font)
        number_color = style_option.palette.color(
            style_option.palette.HighlightedText
            if option.state & QStyle.State_Selected
            else style_option.palette.Text
        )
        painter.setPen(number_color)
        top_rect = content_rect.adjusted(0, 1, 0, -18)
        painter.drawText(top_rect, Qt.AlignLeft | Qt.AlignVCenter, display_text)

        uid_font = style_option.font
        uid_font.setPointSize(max(7, uid_font.pointSize() - 1))
        painter.setFont(uid_font)
        if option.state & QStyle.State_Selected:
            uid_color = style_option.palette.color(style_option.palette.HighlightedText)
        else:
            uid_color = Qt.gray
        painter.setPen(uid_color)
        bottom_rect = content_rect.adjusted(0, 18, 0, 0)
        painter.drawText(bottom_rect, Qt.AlignLeft | Qt.AlignVCenter, uid_text)
        painter.restore()

    def sizeHint(self, option, index):
        base = super().sizeHint(option, index)
        return QSize(base.width(), max(base.height(), 40))
