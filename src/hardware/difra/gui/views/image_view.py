from hardware.difra.gui.image_view_ext.drawing_extension import DrawingMixin
from hardware.difra.gui.image_view_ext.point_editing_extension import (
    PointEditingMixin,
)
from hardware.difra.gui.image_view_ext.zoom_extension import ZoomMixin
from hardware.difra.gui.views.image_view_basic import ImageViewBasic
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QGraphicsEllipseItem, QMenu


class ImageView(ZoomMixin, DrawingMixin, PointEditingMixin, ImageViewBasic):

    def __init__(self, parent=None):
        super().__init__(parent)
        # Initialize drawing functionality.
        self.init_drawing()
        # Initialize point editing functionality.
        self.init_point_editing()
        # rotation_angle is already set in ImageViewBasic's __init__

    def rotate_image(self, angle):
        if self.image_item:
            # Update the rotation of the image_item.
            new_angle = self.image_item.rotation() + angle
            self.image_item.setRotation(new_angle)
            # Keep track of the cumulative rotation.
            self.rotation_angle += angle
        else:
            print("No image to rotate.")

    def delete_selected_shapes(self):
        selected_items = self.scene.selectedItems()
        if not selected_items:
            return

        shapes_to_delete = []
        for shape_info in list(self.shapes):
            shape_item = shape_info.get("item")
            diagonals = shape_info.get("diagonals") or []
            center_marker = shape_info.get("center_marker")
            for item in selected_items:
                if item is self.image_item:
                    continue
                if item is shape_item or item is center_marker or item in diagonals:
                    shapes_to_delete.append(shape_info)
                    break

        for shape_info in shapes_to_delete:
            shape_item = shape_info.get("item")
            if shape_item is not None:
                self.scene.removeItem(shape_item)

            for extra_item in shape_info.get("diagonals") or []:
                self.scene.removeItem(extra_item)

            center_marker = shape_info.get("center_marker")
            if center_marker is not None:
                self.scene.removeItem(center_marker)

            if shape_info in self.shapes:
                self.shapes.remove(shape_info)

        if self.shape_updated_callback:
            self.shape_updated_callback()

    def delete_selected_points(self) -> bool:
        main_window = self.window()
        if main_window is None or not hasattr(main_window, "_request_delete_point_by_id"):
            return False

        selected_items = list(self.scene.selectedItems() or [])
        point_ids = []
        for item in selected_items:
            if item is self.image_item:
                continue
            if not isinstance(item, QGraphicsEllipseItem):
                continue
            try:
                point_type = item.data(0)
                point_id = item.data(1)
            except Exception:
                continue
            if point_type not in ("generated", "user"):
                continue
            if point_id is None:
                continue
            try:
                point_ids.append(int(point_id))
            except Exception:
                continue

        if not point_ids:
            return False

        changed = False
        for point_id in point_ids:
            try:
                changed = bool(main_window._request_delete_point_by_id(point_id)) or changed
            except Exception:
                continue
        return changed

    def keyPressEvent(self, event):
        if event.key() in (Qt.Key_Delete, Qt.Key_Backspace):
            if not self.delete_selected_points():
                self.delete_selected_shapes()
            event.accept()
            return
        super().keyPressEvent(event)

    def contextMenuEvent(self, event):
        selected_items = [item for item in self.scene.selectedItems() if item is not self.image_item]
        if not selected_items:
            super().contextMenuEvent(event)
            return

        menu = QMenu(self)
        delete_points_action = menu.addAction("Delete Selected Point(s)")
        delete_action = menu.addAction("Delete Selected Shape(s)")
        chosen = menu.exec_(event.globalPos())
        if chosen == delete_points_action:
            self.delete_selected_points()
        elif chosen == delete_action:
            self.delete_selected_shapes()
