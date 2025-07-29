from pathlib import Path
import sys

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QApplication, QAction, QFileDialog

from hardware.Ulster.gui.views.main_window_basic import MainWindowBasic
from hardware.Ulster.gui.main_window_ext.drawing_extension import DrawingMixin
from hardware.Ulster.gui.main_window_ext.shape_table_extension import ShapeTableMixin
from hardware.Ulster.gui.main_window_ext.rotation_extension import RotationMixin
from hardware.Ulster.gui.main_window_ext.zone_points_extension import ZonePointsMixin
from hardware.Ulster.gui.main_window_ext.zone_measurements_extension import ZoneMeasurementsMixin
from hardware.Ulster.gui.main_window_ext.state_saver_extension import StateSaverMixin
from hardware.Ulster.gui.main_window_ext.technical_measurements import TechnicalMeasurementsMixin

class MainWindow(RotationMixin, ShapeTableMixin, DrawingMixin,
                 ZonePointsMixin, TechnicalMeasurementsMixin, ZoneMeasurementsMixin, StateSaverMixin,
                 MainWindowBasic):

    def __init__(self, parent=None):
        super().__init__(parent)
        self.measurement_widgets = []
        self.create_shape_table()
        # Create the Zone Measurements widget(right bottom) + Technical .
        self.create_measurements_panel()
        self.create_drawing_actions()
        self.add_drawing_actions_to_tool_bar()
        self.create_delete_action()
        self.add_delete_action_to_tool_bar()
        self.create_rotation_actions()
        self.add_rotation_actions_to_tool_bar()
        # Create the Zone Points widget (left bottom).
        self.create_zone_points_widget()

        self.load_default_masks_and_ponis()
        # Split the bottom dock area horizontally so the panels appear side by side.
        self.splitDockWidget(self.zonePointsDock, self.zoneMeasurementsDock, Qt.Horizontal)
        # Set a callback so that when shapes change, the shape table updates.
        self.image_view.shape_updated_callback = self.update_shape_table

        # Add "Restore State" action to File menu.
        self.add_restore_state_action()
        # Add new "Restore State From File" action to the File menu.
        self.add_restore_state_action_from_file()
        # Add "Save State" button to the toolbar.
        self.add_save_state_action()

    def add_restore_state_action(self):
        restore_state_act = QAction("Restore State", self, triggered=self.restore_state)
        if self.menuBar().actions():
            fileMenu = self.menuBar().actions()[0].menu()
            if fileMenu:
                fileMenu.addAction(restore_state_act)
            else:
                self.menuBar().addAction(restore_state_act)
        else:
            fileMenu = self.menuBar().addMenu("File")
            fileMenu.addAction(restore_state_act)

    def add_restore_state_action_from_file(self):
        restore_state_from_file_act = QAction("Restore State From File", self)
        restore_state_from_file_act.triggered.connect(self.restore_state_from_file)
        if self.menuBar().actions():
            fileMenu = self.menuBar().actions()[0].menu()
            if fileMenu:
                fileMenu.addAction(restore_state_from_file_act)
            else:
                self.menuBar().addAction(restore_state_from_file_act)
        else:
            fileMenu = self.menuBar().addMenu("File")
            fileMenu.addAction(restore_state_from_file_act)

    def restore_state_from_file(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Restore State From File", "", "JSON Files (*.json)")
        if file_path:
            self.restore_state(file_path)

    def add_save_state_action(self):
        save_state_act = QAction("Save State", self, triggered=self.manual_save_state)
        self.toolbar.addAction(save_state_act)



