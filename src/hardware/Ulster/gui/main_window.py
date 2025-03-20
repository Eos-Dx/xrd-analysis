from pathlib import Path
import sys
project_root = Path(__file__).resolve().parent.parent.parent.parent
print(project_root)
sys.path.insert(0, str(project_root))

from PyQt5.QtWidgets import QApplication
from hardware.Ulster.gui.main_window_basic import MainWindowBasic
from hardware.Ulster.gui.main_window_ext.drawing_extension import DrawingMixin
from hardware.Ulster.gui.main_window_ext.shape_table_extension import ShapeTableMixin
from hardware.Ulster.gui.main_window_ext.rotation_extension import RotationMixin
from hardware.Ulster.gui.main_window_ext.zone_points_extension import ZonePointsMixin


class MainWindow(RotationMixin, ShapeTableMixin, DrawingMixin, ZonePointsMixin, MainWindowBasic):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.createShapeTable()
        self.createDrawingActions()
        self.addDrawingActionsToToolBar()
        self.createDeleteAction()
        self.addDeleteActionToToolBar()
        self.createRotationActions()
        self.addRotationActionsToToolBar()
        # Create the Zone Points widget.
        self.createZonePointsWidget()
        # Set a callback so that when shapes change, the table updates.
        self.image_view.shapeUpdatedCallback = self.updateShapeTable

if __name__ == '__main__':
    import sys
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())
