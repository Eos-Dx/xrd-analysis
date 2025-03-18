from hardware.Ulster.gui.image_view_basic import ImageViewBasic
from hardware.Ulster.gui.image_view_ext.zoom_extension import ZoomMixin
from hardware.Ulster.gui.image_view_ext.drawing_extension import DrawingMixin

class ImageView(ZoomMixin, DrawingMixin, ImageViewBasic):
    def __init__(self, parent=None):
        super().__init__(parent)
        # Initialize drawing functionality from the drawing mixin.
        self.initDrawing()
