from PyQt5.QtWidgets import QGraphicsView, QGraphicsScene
from PyQt5.QtCore import QRectF
from PyQt5.QtGui import QPixmap

class ImageViewBasic(QGraphicsView):
    def __init__(self, parent=None):
        self.scene = QGraphicsScene()
        super().__init__(self.scene, parent)
        self.image_item = None
        self.current_pixmap = None
        # New attributes to store state:
        self.current_image_path = None
        self.rotation_angle = 0

    def setImage(self, pixmap: QPixmap, image_path=None):
        self.current_pixmap = pixmap
        # Store the file path if provided.
        if image_path:
            self.current_image_path = image_path
        self.scene.clear()
        self.image_item = self.scene.addPixmap(pixmap)
        # Reset rotation angle when a new image is set.
        self.rotation_angle = 0
        # Compute the image rectangle.
        imageRect = QRectF(pixmap.rect())
        # Map the current viewport rectangle to scene coordinates.
        viewRect = self.mapToScene(self.viewport().rect()).boundingRect()
        # Expand the scene rect to be the union of the image and the view.
        combinedRect = imageRect.united(viewRect)
        self.setSceneRect(combinedRect)
