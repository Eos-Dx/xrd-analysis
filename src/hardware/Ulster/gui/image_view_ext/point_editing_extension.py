from PyQt5.QtWidgets import QGraphicsEllipseItem
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPen, QColor


class PointEditingMixin:
    def initPointEditing(self):
        # Initialize container for user-added points if not present.
        if not hasattr(self, 'user_points'):
            self.user_points = []

    def mouseDoubleClickEvent(self, event):
        if event.button() == Qt.LeftButton:
            # Left double-click: add point if inside include zone.
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
            radius = 3
            pt_item = QGraphicsEllipseItem(pos.x() - radius, pos.y() - radius, 2 * radius, 2 * radius)
            pt_item.setBrush(QColor("blue"))
            pt_item.setPen(QPen(Qt.NoPen))
            pt_item.setFlags(QGraphicsEllipseItem.ItemIsSelectable | QGraphicsEllipseItem.ItemIsMovable)
            self.scene.addItem(pt_item)
            self.user_points.append(pt_item)
            self.scene.update()
        elif event.button() == Qt.RightButton:
            # Right double-click: delete point if near the click.
            pos = self.mapToScene(event.pos())
            threshold = 5  # pixels
            for pt in list(self.user_points):  # iterate over a copy
                rect = pt.rect()
                center = pt.mapToScene(rect.center())
                dx = pos.x() - center.x()
                dy = pos.y() - center.y()
                if (dx * dx + dy * dy) ** 0.5 < threshold:
                    self.scene.removeItem(pt)
                    self.user_points.remove(pt)
                    self.scene.update()
                    break
        else:
            super().mouseDoubleClickEvent(event)
