from PyQt5.QtWidgets import QGraphicsEllipseItem
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPen, QColor
from hardware.Ulster.gui.extra.elements import HoverableEllipseItem

class PointEditingMixin:
    def initPointEditing(self):
        # Initialize container for user-added points if not present.
        if not hasattr(self, 'user_points'):
            self.user_points = []
        # Also initialize a list for associated user-defined zones.
        if not hasattr(self, "user_defined_zones"):
            self.user_defined_zones = []

    def mouseDoubleClickEvent(self, event):
        if event.button() == Qt.LeftButton:
            pos = self.mapToScene(event.pos())
            include_shape = None
            for shape in self.shapes:
                if shape.get("role", "include") == "include":
                    include_shape = shape["item"]
                    break
            if include_shape is None:
                print("No include zone defined.")
                return
            if not include_shape.contains(include_shape.mapFromScene(pos)):
                print("Double-click point is not inside the include zone.")
                return

            # Define the zone radius for the shade zone.
            zone_radius = 10  # Adjust as needed.
            # Create the shade zone (a circle behind the point).
            zone_item = QGraphicsEllipseItem(pos.x() - zone_radius, pos.y() - zone_radius,
                                             2 * zone_radius, 2 * zone_radius)
            zone_color = QColor("blue")
            zone_color.setAlphaF(0.2)
            zone_item.setBrush(zone_color)
            zone_item.setPen(QPen(Qt.NoPen))
            self.scene.addItem(zone_item)
            # Save the zone in a dedicated list (using self instead of self.image_view).
            if not hasattr(self, "user_defined_zones"):
                self.user_defined_zones = []
            self.user_defined_zones.append(zone_item)

            # Create the user-defined point marker as a HoverableEllipseItem.
            marker_radius = 5  # Make the marker a bit larger.
            pt_item = HoverableEllipseItem(pos.x() - marker_radius, pos.y() - marker_radius,
                                           2 * marker_radius, 2 * marker_radius)
            pt_item.setBrush(QColor("blue"))
            pt_item.setPen(QPen(Qt.NoPen))
            pt_item.setFlags(QGraphicsEllipseItem.ItemIsSelectable | QGraphicsEllipseItem.ItemIsMovable)
            pt_item.setData(0, "user")
            # Set the hover callback so the zone can be highlighted.
            main_window = self.window()  # Get the top-level window.
            pt_item.hoverCallback = getattr(main_window, "pointHoverChanged", lambda item, hovered: None)
            self.scene.addItem(pt_item)
            self.user_points.append(pt_item)
            self.scene.update()
        elif event.button() == Qt.RightButton:
            pos = self.mapToScene(event.pos())
            threshold = 5  # pixels
            for pt in list(self.user_points):  # iterate over a copy
                rect = pt.rect()
                center = pt.mapToScene(rect.center())
                dx = pos.x() - center.x()
                dy = pos.y() - center.y()
                if (dx * dx + dy * dy) ** 0.5 < threshold:
                    # Also remove associated zone if available.
                    idx = self.user_points.index(pt)
                    if hasattr(self, "user_defined_zones") and idx < len(self.user_defined_zones):
                        zone_item = self.user_defined_zones.pop(idx)
                        self.scene.removeItem(zone_item)
                    self.scene.removeItem(pt)
                    self.user_points.remove(pt)
                    self.scene.update()
                    break
        else:
            super().mouseDoubleClickEvent(event)
