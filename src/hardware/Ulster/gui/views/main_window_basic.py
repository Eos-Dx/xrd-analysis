import os
import json
from pathlib import Path
from PyQt5.QtWidgets import QMainWindow, QAction, QFileDialog, QToolBar
from PyQt5.QtGui import QPixmap, QIcon
from hardware.Ulster.gui.views.image_view import ImageView


class MainWindowBasic(QMainWindow):

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("EosDX Scanning Software")
        # Set the window icon using the logo image.
        logo_path = Path(__file__).resolve().parent.parent.parent / 'resources/images/rick_final.png'
        if logo_path.exists():
            self.setWindowIcon(QIcon(str(logo_path)))
        else:
            print("Logo file not found:", logo_path)
        self.resize(800, 600)
        self.config = self.load_config()
        self.image_view = ImageView(self)
        self.setCentralWidget(self.image_view)
        self.create_actions()
        self.create_menus()
        self.create_tool_bar()
        self.check_dev_mode()

    def load_config(self):
        config_path = Path(__file__).resolve().parent.parent.parent / 'resources/config/main.json'
        try:
            with open(config_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            print("Error loading config:", e)
            return {}

    def create_actions(self):
        self.open_act = QAction("Open Image", self, triggered=self.open_image)

    def create_menus(self):
        fileMenu = self.menuBar().addMenu("File")
        fileMenu.addAction(self.open_act)

    def create_tool_bar(self):
        # Store the toolbar so extensions can add actions.
        self.toolbar = QToolBar("Tools", self)
        self.addToolBar(self.toolbar)
        self.toolbar.addAction(self.open_act)

    def open_image(self):
        default_folder = self.config.get("default_image_folder", "")
        self.image_path, _ = QFileDialog.getOpenFileName(self,
                                                         "Open Image",
                                                         default_folder,
                                                         "Image Files (*.png *.jpg *.jpeg);;All Files (*)"
                                                         )
        if self.image_path:
            pixmap = QPixmap(self.image_path)
            self.image_view.set_image(pixmap, image_path=self.image_path)
            try:
                self.delete_all_shapes_from_table()
                self.delete_all_points()
            except Exception as e:
                print(e)

    def check_dev_mode(self):
        # If DEV mode is enabled, automatically load the default image.
        if self.config.get("DEV", False):
            default_image = self.config.get("default_image", "")
            if default_image and os.path.exists(default_image):
                pixmap = QPixmap(default_image)
                self.image_view.set_image(pixmap, image_path=default_image)
            else:
                print("Default image file not found:", default_image)
