from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QGraphicsView

class ZoomMixin:
    def wheelEvent(self, event):
        zoomInFactor = 1.25
        zoomOutFactor = 1 / zoomInFactor
        if event.angleDelta().y() > 0:
            factor = zoomInFactor
        else:
            factor = zoomOutFactor
        self.scale(factor, factor)

    def mousePressEvent(self, event):
        if event.button() == Qt.MiddleButton:
            # Store the initial position for dragging and change the cursor.
            self._dragPos = event.pos()
            self.setCursor(Qt.ClosedHandCursor)
            event.accept()
        else:
            super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if event.buttons() & Qt.MiddleButton and hasattr(self, '_dragPos'):
            # Calculate the delta movement.
            delta = event.pos() - self._dragPos
            self._dragPos = event.pos()
            # Pan the view by updating the scrollbars.
            self.horizontalScrollBar().setValue(self.horizontalScrollBar().value() - delta.x())
            self.verticalScrollBar().setValue(self.verticalScrollBar().value() - delta.y())
            event.accept()
        else:
            super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MiddleButton:
            self.unsetCursor()
            if hasattr(self, '_dragPos'):
                del self._dragPos
            event.accept()
        else:
            super().mouseReleaseEvent(event)
