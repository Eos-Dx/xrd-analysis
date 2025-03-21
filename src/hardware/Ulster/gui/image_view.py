from hardware.Ulster.gui.image_view_basic import ImageViewBasic
from hardware.Ulster.gui.image_view_ext.zoom_extension import ZoomMixin
from hardware.Ulster.gui.image_view_ext.drawing_extension import DrawingMixin
from hardware.Ulster.gui.image_view_ext.point_editing_extension import PointEditingMixin

class ImageView(ZoomMixin, DrawingMixin, PointEditingMixin, ImageViewBasic):
    def __init__(self, parent=None):
        super().__init__(parent)
        # Initialize drawing functionality.
        self.initDrawing()
        # Initialize point editing functionality.
        self.initPointEditing()
        # rotation_angle is already set in ImageViewBasic's __init__

    def rotateImage(self, angle):
        if self.image_item:
            # Update the rotation of the image_item.
            new_angle = self.image_item.rotation() + angle
            self.image_item.setRotation(new_angle)
            # Keep track of the cumulative rotation.
            self.rotation_angle += angle
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
