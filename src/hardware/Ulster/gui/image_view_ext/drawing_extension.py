from PyQt5.QtCore import QRectF
from PyQt5.QtWidgets import QGraphicsRectItem, QGraphicsEllipseItem, QGraphicsItem

class DrawingMixin:
    def initDrawing(self):
        # Initialize drawing properties.
        self.drawing_mode = None  # "rect", "ellipse", "crop", or None (select mode)
        self.pen = self._createPen()
        self.start_point = None
        self.current_shape = None
        self.shapes = []         # List to hold drawn shapes (each: {"id", "type", "item"})
        self.shape_counter = 1
        self.shapeUpdatedCallback = None

    def _createPen(self):
        from PyQt5.QtGui import QPen
        from PyQt5.QtCore import Qt
        return QPen(Qt.red, 2, Qt.DashLine)

    def setDrawingMode(self, mode):
        self.drawing_mode = mode

    def mousePressEvent(self, event):
        if self.drawing_mode in ["rect", "ellipse", "crop"]:
            # Do nothing if no image is loaded.
            if self.current_pixmap is None:
                print("No image loaded. Please open an image first.")
                return
            self.start_point = self.mapToScene(event.pos())
            rect = QRectF(self.start_point, self.start_point)
            if self.drawing_mode in ["rect", "crop"]:
                rect_item = QGraphicsRectItem(rect)
                rect_item.setPen(self.pen)
                if self.drawing_mode == "rect":
                    rect_item.setFlags(QGraphicsItem.ItemIsSelectable | QGraphicsItem.ItemIsMovable)
                self.current_shape = rect_item
                self.scene.addItem(rect_item)
            elif self.drawing_mode == "ellipse":
                ellipse_item = QGraphicsEllipseItem(rect)
                ellipse_item.setPen(self.pen)
                ellipse_item.setFlags(QGraphicsItem.ItemIsSelectable | QGraphicsItem.ItemIsMovable)
                self.current_shape = ellipse_item
                self.scene.addItem(ellipse_item)
        else:
            super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self.drawing_mode and self.start_point and self.current_shape:
            current_point = self.mapToScene(event.pos())
            rect = QRectF(self.start_point, current_point).normalized()
            if hasattr(self.current_shape, 'setRect'):
                self.current_shape.setRect(rect)
        else:
            super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if self.drawing_mode:
            if self.drawing_mode == "crop":
                if self.current_shape and self.current_pixmap:
                    rect = self.current_shape.rect().normalized()
                    from PyQt5.QtCore import QRectF
                    image_rect = QRectF(self.current_pixmap.rect())
                    crop_rect = rect.intersected(image_rect)
                    # Remove the crop rectangle BEFORE applying the crop.
                    self.scene.removeItem(self.current_shape)
                    self.current_shape = None
                    if crop_rect.width() > 0 and crop_rect.height() > 0:
                        cropped_pixmap = self.current_pixmap.copy(
                            int(crop_rect.x()),
                            int(crop_rect.y()),
                            int(crop_rect.width()),
                            int(crop_rect.height())
                        )
                        self.setImage(cropped_pixmap)
                    else:
                        print("Invalid crop rectangle.")
            elif self.drawing_mode in ["rect", "ellipse"]:
                if self.current_shape:
                    shape_info = {
                        "id": self.shape_counter,
                        "type": "Rectangle" if self.drawing_mode == "rect" else "Circle",
                        "item": self.current_shape,
                        "role": "include"  # Default role is include.
                    }
                    self.shapes.append(shape_info)
                    self.shape_counter += 1
                    if self.shapeUpdatedCallback:
                        self.shapeUpdatedCallback()
            self.start_point = None
            self.current_shape = None
        else:
            super().mouseReleaseEvent(event)
