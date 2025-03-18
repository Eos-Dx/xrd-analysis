# hardware/Ulster/gui/image_view_ext/zoom_extension.py
class ZoomMixin:
    def wheelEvent(self, event):
        zoomInFactor = 1.25
        zoomOutFactor = 1 / zoomInFactor
        if event.angleDelta().y() > 0:
            factor = zoomInFactor
        else:
            factor = zoomOutFactor
        self.scale(factor, factor)
