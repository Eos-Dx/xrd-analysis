from hardware.Ulster.gui.image_view_basic import ImageViewBasic
from hardware.Ulster.gui.image_view_ext.zoom_extension import ZoomMixin
from hardware.Ulster.gui.image_view_ext.drawing_extension import DrawingMixin


class ImageView(ZoomMixin, DrawingMixin, ImageViewBasic):
    def __init__(self, parent=None):
        super().__init__(parent)
        # Initialize drawing functionality.
        self.initDrawing()

    def rotateImage(self, angle):
        if self.image_item:
            self.image_item.setRotation(self.image_item.rotation() + angle)
        else:
            print("No image to rotate.")

    def deleteSelectedShapes(self):
        selected_items = self.scene.selectedItems()
        for item in selected_items:
            if item is not self.image_item:
                self.scene.removeItem(item)
                self.shapes = [s for s in self.shapes if s["item"] != item]
        if self.shapeUpdatedCallback:
            self.shapeUpdatedCallback()
