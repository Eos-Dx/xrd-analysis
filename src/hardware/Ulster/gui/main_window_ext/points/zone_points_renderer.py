"""Renderer and table manager for zone points functionality."""

from typing import Any, Dict, List, Optional, Tuple

from core.geometry.constants import ZonePointsConstants
from gui.extra.elements import HoverableEllipseItem
from PyQt5 import sip
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QColor, QPen
from PyQt5.QtWidgets import QGraphicsEllipseItem, QTableWidgetItem
from utils.logging import get_module_logger

logger = get_module_logger(__name__)


class ZonePointsRenderer:
    """Handles rendering of zone points and zones."""

    @staticmethod
    def create_zone_item(x: float, y: float, radius: float) -> QGraphicsEllipseItem:
        """Create a zone (cyan circle) item."""
        item = QGraphicsEllipseItem(x - radius, y - radius, 2 * radius, 2 * radius)
        cyan_color = QColor(ZonePointsConstants.ZONE_COLOR)
        cyan_color.setAlphaF(ZonePointsConstants.ZONE_ALPHA)
        item.setBrush(cyan_color)
        item.setPen(QPen(Qt.NoPen))
        return item

    @staticmethod
    def create_point_item(
        x: float, y: float, point_id: int, point_type: str = "generated"
    ) -> HoverableEllipseItem:
        """Create a point (red/blue dot) item."""
        radius = ZonePointsConstants.POINT_RADIUS
        item = HoverableEllipseItem(
            x - radius,
            y - radius,
            ZonePointsConstants.POINT_DIAMETER,
            ZonePointsConstants.POINT_DIAMETER,
        )

        color = (
            ZonePointsConstants.POINT_COLOR_GENERATED
            if point_type == "generated"
            else ZonePointsConstants.POINT_COLOR_USER
        )
        item.setBrush(QColor(color))
        item.setPen(QPen(Qt.NoPen))
        item.setFlags(
            QGraphicsEllipseItem.ItemIsSelectable | QGraphicsEllipseItem.ItemIsMovable
        )
        item.setData(0, point_type)
        item.setData(1, point_id)

        return item


class ZonePointsTableManager:
    """Manages table updates for zone points (no embedded measurement widgets)."""

    @staticmethod
    def build_points_snapshot(
        image_view,
    ) -> List[Tuple[float, float, str, Optional[int]]]:
        """Build a snapshot of all current points with their data."""
        points = []

        # Generated points
        for item in image_view.points_dict["generated"]["points"]:
            if sip.isdeleted(item):
                continue
            c = item.sceneBoundingRect().center()
            pid = item.data(1)
            points.append(
                (c.x(), c.y(), "generated", int(pid) if pid is not None else None)
            )

        # User points
        for item in image_view.points_dict["user"]["points"]:
            if sip.isdeleted(item):
                continue
            c = item.sceneBoundingRect().center()
            pid = item.data(1)
            points.append((c.x(), c.y(), "user", int(pid) if pid is not None else None))

        return points

    @staticmethod
    def cleanup_deleted_widgets(
        measurement_widgets: Dict[int, Any],
        points: List[Tuple[float, float, str, Optional[int]]],
    ):
        """Clean up measurement widgets for points that no longer exist."""
        current_point_ids = {pid for (_, _, _, pid) in points if pid is not None}

        # Remove widgets for deleted points
        for pid in list(measurement_widgets.keys()):
            if pid not in current_point_ids:
                widget = measurement_widgets.pop(pid)
                if widget and not sip.isdeleted(widget):
                    widget.setParent(None)
                    widget.deleteLater()
                    logger.debug(
                        "Cleaned up measurement widget for deleted point", point_id=pid
                    )

    @staticmethod
    def populate_table_rows(
        points_table,
        points: List[Tuple[float, float, str, Optional[int]]],
        measurement_widgets: Dict[int, Any],
        real_x_pos_mm,
        real_y_pos_mm,
        include_center: Tuple[float, float],
        pixel_to_mm_ratio: float,
    ):
        """Populate table rows with point data (no embedded measurement widgets)."""
        for idx, (x, y, ptype, point_id) in enumerate(points):
            # Set basic point data
            points_table.setItem(
                idx, 0, QTableWidgetItem("" if point_id is None else str(point_id))
            )
            points_table.setItem(idx, 1, QTableWidgetItem(f"{x:.2f}"))
            points_table.setItem(idx, 2, QTableWidgetItem(f"{y:.2f}"))

            # Set coordinate data
            if pixel_to_mm_ratio:
                x_mm = (
                    real_x_pos_mm.value() - (x - include_center[0]) / pixel_to_mm_ratio
                )
                y_mm = (
                    real_y_pos_mm.value() - (y - include_center[1]) / pixel_to_mm_ratio
                )
                points_table.setItem(idx, 3, QTableWidgetItem(f"{x_mm:.2f}"))
                points_table.setItem(idx, 4, QTableWidgetItem(f"{y_mm:.2f}"))
            else:
                points_table.setItem(idx, 3, QTableWidgetItem("N/A"))
                points_table.setItem(idx, 4, QTableWidgetItem("N/A"))

            # No measurement widget embedding in the table anymore
            pass
