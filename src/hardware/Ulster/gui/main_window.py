# gui/main_window.py
import sys
import os
import json
from PyQt5.QtWidgets import (QMainWindow, QAction, QActionGroup, QFileDialog,
                             QToolBar, QApplication, QDockWidget, QTableWidget,
                             QTableWidgetItem, QAbstractItemView)
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import Qt
from hardware.Ulster.gui.image_view import ImageView


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Scan Area Selector")
        self.resize(1000, 700)

        # Load configuration from JSON file.
        self.config = self.load_config()

        # Create and set the central widget.
        self.image_view = ImageView(self)
        self.setCentralWidget(self.image_view)

        # Create UI components.
        self.createActions()
        self.createMenus()
        self.createToolBar()
        self.createDockTable()

        # DEV mode: if enabled, auto-load a specific image.
        if self.config.get("DEV", False):
            dev_image_path = r'C:\dev\eos_play\unzipped_data\Ulster\pancreas\sheep_pancreas\U2\Xena_Cu\20250313\P1a_L4_20250218.jpg'
            if os.path.exists(dev_image_path):
                pixmap = QPixmap(dev_image_path)
                self.image_view.setImage(pixmap)
            else:
                print("Dev image file not found:", dev_image_path)

        # Whenever shapes are updated in the view, refresh the table.
        self.image_view.shapeUpdatedCallback = self.updateShapesTable

    def load_config(self):
        config_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            '..',
            'resources',
            'config',
            'main.json'
        )
        config = {}
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
        except Exception as e:
            print(f"Could not load config file at {config_path}. Error: {e}")
        return config

    def createActions(self):
        self.openAct = QAction("Open Image", self, triggered=self.openImage)
        self.selectRectAct = QAction("Rectangle", self, checkable=True, triggered=self.selectRectMode)
        self.selectEllipseAct = QAction("Circle", self, checkable=True, triggered=self.selectEllipseMode)
        self.cropAct = QAction("Crop", self, checkable=True, triggered=self.selectCropMode)
        self.selectAct = QAction("Select", self, checkable=True, triggered=self.selectSelectMode)

        # Group actions so that only one mode is active.
        self.modeGroup = QActionGroup(self)
        self.modeGroup.addAction(self.selectRectAct)
        self.modeGroup.addAction(self.selectEllipseAct)
        self.modeGroup.addAction(self.cropAct)
        self.modeGroup.addAction(self.selectAct)

        # Default to select (arrow) mode.
        self.selectAct.setChecked(True)

    def createMenus(self):
        fileMenu = self.menuBar().addMenu("File")
        fileMenu.addAction(self.openAct)

    def createToolBar(self):
        toolbar = QToolBar("Tools", self)
        self.addToolBar(toolbar)
        toolbar.addAction(self.selectRectAct)
        toolbar.addAction(self.selectEllipseAct)
        toolbar.addAction(self.cropAct)
        toolbar.addAction(self.selectAct)

    def createDockTable(self):
        # Create a dock widget to hold the shapes table.
        self.dock = QDockWidget("Shapes", self)
        self.table = QTableWidget(0, 6, self)
        self.table.setHorizontalHeaderLabels(["ID", "Type", "X", "Y", "Width", "Height"])
        self.table.setEditTriggers(QAbstractItemView.DoubleClicked | QAbstractItemView.SelectedClicked)
        self.table.cellChanged.connect(self.on_table_cell_changed)
        self.dock.setWidget(self.table)
        self.addDockWidget(Qt.RightDockWidgetArea, self.dock)

    def updateShapesTable(self):
        # Update the table from the shapes stored in image_view.
        shapes = self.image_view.shapes  # list of dicts: {"id", "type", "item"}
        self.table.blockSignals(True)  # Prevent cellChanged signals during update.
        self.table.setRowCount(len(shapes))
        for row, shape_info in enumerate(shapes):
            shape_id = shape_info["id"]
            shape_type = shape_info["type"]
            item = shape_info["item"]
            # Retrieve geometry; use item.rect() if available.
            rect = item.rect() if hasattr(item, 'rect') else item.boundingRect()
            self.table.setItem(row, 0, QTableWidgetItem(str(shape_id)))
            self.table.setItem(row, 1, QTableWidgetItem(shape_type))
            self.table.setItem(row, 2, QTableWidgetItem(f"{rect.x():.2f}"))
            self.table.setItem(row, 3, QTableWidgetItem(f"{rect.y():.2f}"))
            self.table.setItem(row, 4, QTableWidgetItem(f"{rect.width():.2f}"))
            self.table.setItem(row, 5, QTableWidgetItem(f"{rect.height():.2f}"))
        self.table.blockSignals(False)

    def on_table_cell_changed(self, row, column):
        # When a cell is edited, update the corresponding shape's geometry.
        try:
            shape_id = int(self.table.item(row, 0).text())
            for shape_info in self.image_view.shapes:
                if shape_info["id"] == shape_id:
                    item = shape_info["item"]
                    x = float(self.table.item(row, 2).text())
                    y = float(self.table.item(row, 3).text())
                    w = float(self.table.item(row, 4).text())
                    h = float(self.table.item(row, 5).text())
                    if hasattr(item, 'setRect'):
                        item.setRect(x, y, w, h)
                    break
        except Exception as e:
            print("Error updating shape from table:", e)

    def openImage(self):
        default_folder = self.config.get("default_image_folder", "")
        options = QFileDialog.Options()
        fileName, _ = QFileDialog.getOpenFileName(
            self,
            "Open Image",
            default_folder,
            "Image Files (*.png *.jpg *.jpeg);;PNG Files (*.png);;JPEG Files (*.jpg *.jpeg);;All Files (*)",
            options=options
        )
        if fileName:
            pixmap = QPixmap(fileName)
            self.image_view.setImage(pixmap)
            self.updateShapesTable()

    def selectRectMode(self):
        self.image_view.setDrawingMode("rect")

    def selectEllipseMode(self):
        self.image_view.setDrawingMode("ellipse")

    def selectCropMode(self):
        self.image_view.setDrawingMode("crop")

    def selectSelectMode(self):
        # Switch off drawing; allow selection and moving of existing shapes.
        self.image_view.setDrawingMode(None)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())
