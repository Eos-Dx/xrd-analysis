from PyQt5.QtWidgets import QGraphicsView, QGraphicsScene
from PyQt5.QtCore import QRectF
from PyQt5.QtGui import QPixmap


class ImageViewBasic(QGraphicsView):
    def __init__(self, parent=None):
        self.scene = QGraphicsScene()
        super().__init__(self.scene, parent)
        self.image_item = None
        self.current_pixmap = None

    def setImage(self, pixmap: QPixmap):
        self.current_pixmap = pixmap
        self.scene.clear()
        self.image_item = self.scene.addPixmap(pixmap)
        # Compute the image rectangle
        imageRect = QRectF(pixmap.rect())
        # Map the current viewport rectangle to scene coordinates.
        viewRect = self.mapToScene(self.viewport().rect()).boundingRect()
        # Expand the scene rect to be the union of the image and the view.
        combinedRect = imageRect.united(viewRect)
        self.setSceneRect(combinedRect)
