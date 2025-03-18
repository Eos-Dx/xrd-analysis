# hardware/Ulster/gui/main_window.py
from PyQt5.QtWidgets import QApplication
from hardware.Ulster.gui.main_window_basic import MainWindowBasic
from hardware.Ulster.gui.main_window_ext.shape_table_extension import ShapeTableMixin
from hardware.Ulster.gui.main_window_ext.drawing_extension import DrawingMixin

class MainWindow(ShapeTableMixin, DrawingMixin, MainWindowBasic):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.createShapeTable()
        self.createDrawingActions()
        self.addDrawingActionsToToolBar()
        # Set callback so that when shapes change in the image view, the table updates.
        self.image_view.shapeUpdatedCallback = self.updateShapeTable

if __name__ == '__main__':
    import sys
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())
