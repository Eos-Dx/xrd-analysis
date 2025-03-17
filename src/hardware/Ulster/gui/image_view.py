# gui/image_view.py
from PyQt5.QtWidgets import QGraphicsView, QGraphicsScene, QGraphicsItem, QGraphicsRectItem, QGraphicsEllipseItem
from PyQt5.QtGui import QPen, QPixmap
from PyQt5.QtCore import Qt, QRectF

class ImageView(QGraphicsView):
    def __init__(self, parent=None):
        self.scene = QGraphicsScene()
        super().__init__(self.scene, parent)
        self.image_item = None
        self.current_pixmap = None
        self.start_point = None
        self.current_shape = None
        self.drawing_mode = None  # "rect", "ellipse", "crop" or None (select)
        self.pen = QPen(Qt.red, 2, Qt.DashLine)
        self.shapes = []  # List of shapes: each is a dict with keys "id", "type", "item"
        self.shape_counter = 1
        self.shapeUpdatedCallback = None  # Callback to refresh table in main window

    def setImage(self, pixmap):
        self.current_pixmap = pixmap
        self.scene.clear()
        self.image_item = self.scene.addPixmap(pixmap)
        self.setSceneRect(QRectF(pixmap.rect()))
        self.shapes = []  # Reset any drawn shapes.
        if self.shapeUpdatedCallback:
            self.shapeUpdatedCallback()

    def setDrawingMode(self, mode):
        self.drawing_mode = mode

    def wheelEvent(self, event):
        zoomInFactor = 1.25
        zoomOutFactor = 1 / zoomInFactor
        if event.angleDelta().y() > 0:
            zoomFactor = zoomInFactor
        else:
            zoomFactor = zoomOutFactor
        self.scale(zoomFactor, zoomFactor)

    def mousePressEvent(self, event):
        if self.drawing_mode in ["rect", "ellipse", "crop"]:
            self.start_point = self.mapToScene(event.pos())
            rect = QRectF(self.start_point, self.start_point)
            if self.drawing_mode in ["rect", "crop"]:
                rect_item = QGraphicsRectItem(rect)
                rect_item.setPen(self.pen)
                # In rectangle mode (annotation) allow selection and movement.
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
            # In select (arrow) mode, let QGraphicsView handle selection.
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
                if self.current_shape:
                    rect = self.current_shape.rect().normalized()
                    x = int(rect.x())
                    y = int(rect.y())
                    w = int(rect.width())
                    h = int(rect.height())
                    if w > 0 and h > 0:
                        cropped_pixmap = self.current_pixmap.copy(x, y, w, h)
                        self.setImage(cropped_pixmap)
                    else:
                        print("Invalid crop rectangle.")
                    self.scene.removeItem(self.current_shape)
            elif self.drawing_mode in ["rect", "ellipse"]:
                if self.current_shape:
                    # Finalize the shape and add it to the shapes list.
                    shape_info = {
                        "id": self.shape_counter,
                        "type": "Rectangle" if self.drawing_mode == "rect" else "Circle",
                        "item": self.current_shape
                    }
                    self.shapes.append(shape_info)
                    self.shape_counter += 1
                    if self.shapeUpdatedCallback:
                        self.shapeUpdatedCallback()
            self.start_point = None
            self.current_shape = None
        else:
            super().mouseReleaseEvent(event)
