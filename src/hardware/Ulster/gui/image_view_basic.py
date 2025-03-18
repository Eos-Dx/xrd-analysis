# hardware/Ulster/gui/image_view_basic.py
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
        self.setSceneRect(QRectF(pixmap.rect()))
