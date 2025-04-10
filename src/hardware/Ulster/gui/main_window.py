from pathlib import Path
import sys

# Set the project root.
project_root = Path(__file__).resolve().parent.parent.parent.parent
print(project_root)
sys.path.insert(0, str(project_root))

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QApplication, QAction, QFileDialog
from hardware.Ulster.gui.main_window_basic import MainWindowBasic
from hardware.Ulster.gui.main_window_ext.drawing_extension import DrawingMixin
from hardware.Ulster.gui.main_window_ext.shape_table_extension import ShapeTableMixin
from hardware.Ulster.gui.main_window_ext.rotation_extension import RotationMixin
from hardware.Ulster.gui.main_window_ext.zone_points_extension import ZonePointsMixin
from hardware.Ulster.gui.main_window_ext.zone_measurements_extension import ZoneMeasurementsMixin
from hardware.Ulster.gui.main_window_ext.state_saver_extension import StateSaverMixin


class MainWindow(RotationMixin, ShapeTableMixin, DrawingMixin,
                 ZonePointsMixin, ZoneMeasurementsMixin, StateSaverMixin,
                 MainWindowBasic):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.createShapeTable()
        self.createDrawingActions()
        self.addDrawingActionsToToolBar()
        self.createDeleteAction()
        self.addDeleteActionToToolBar()
        self.createRotationActions()
        self.addRotationActionsToToolBar()
        # Create the Zone Points widget (left bottom).
        self.createZonePointsWidget()
        # Create the Zone Measurements widget (right bottom).
        self.createZoneMeasurementsWidget()
        # Split the bottom dock area horizontally so the panels appear side by side.
        self.splitDockWidget(self.zonePointsDock, self.zoneMeasurementsDock, Qt.Horizontal)
        # Set a callback so that when shapes change, the shape table updates.
        self.image_view.shapeUpdatedCallback = self.updateShapeTable

        # Add "Restore State" action to File menu.
        self.addRestoreStateAction()
        # Add new "Restore State From File" action to the File menu.
        self.addRestoreStateActionFromFile()
        # Add "Save State" button to the toolbar.
        self.addSaveStateAction()

    def addRestoreStateAction(self):
        restoreStateAct = QAction("Restore State", self, triggered=self.restoreState)
        if self.menuBar().actions():
            fileMenu = self.menuBar().actions()[0].menu()
            if fileMenu:
                fileMenu.addAction(restoreStateAct)
            else:
                self.menuBar().addAction(restoreStateAct)
        else:
            fileMenu = self.menuBar().addMenu("File")
            fileMenu.addAction(restoreStateAct)

    def addRestoreStateActionFromFile(self):
        restoreStateFromFileAct = QAction("Restore State From File", self)
        restoreStateFromFileAct.triggered.connect(self.restoreStateFromFile)
        if self.menuBar().actions():
            fileMenu = self.menuBar().actions()[0].menu()
            if fileMenu:
                fileMenu.addAction(restoreStateFromFileAct)
            else:
                self.menuBar().addAction(restoreStateFromFileAct)
        else:
            fileMenu = self.menuBar().addMenu("File")
            fileMenu.addAction(restoreStateFromFileAct)

    def restoreStateFromFile(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Restore State From File", "", "JSON Files (*.json)")
        if file_path:
            self.restoreState(file_path)

    def addSaveStateAction(self):
        saveStateAct = QAction("Save State", self, triggered=self.manualSaveState)
        self.toolBar.addAction(saveStateAct)


if __name__ == '__main__':
    import os
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())
