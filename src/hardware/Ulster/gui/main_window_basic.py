import os
import json
from pathlib import Path
from PyQt5.QtWidgets import QMainWindow, QAction, QFileDialog, QToolBar
from PyQt5.QtGui import QPixmap, QIcon
from hardware.Ulster.gui.image_view import ImageView


class MainWindowBasic(QMainWindow):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("EosDX Scanning Software")
        # Set the window icon using the logo image.
        logo_path = Path('..har /Ulster/resources/images/rick_final.png')  # Adjust the path.
        if logo_path.exists():
            self.setWindowIcon(QIcon(str(logo_path)))
        else:
            print("Logo file not found:", logo_path)
        self.resize(800, 600)
        self.config = self.load_config()
        self.image_view = ImageView(self)
        self.setCentralWidget(self.image_view)
        self.createActions()
        self.createMenus()
        self.createToolBar()
        self.checkDevMode()

    def load_config(self):
        # Adjust the path as needed.
        config_path = Path('C:/dev/xrd-analysis/src/hardware/Ulster/resources/config/main.json')  # Adjust the path.
        try:
            with open(config_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            print("Error loading config:", e)
            return {}

    def createActions(self):
        self.openAct = QAction("Open Image", self, triggered=self.openImage)

    def createMenus(self):
        fileMenu = self.menuBar().addMenu("File")
        fileMenu.addAction(self.openAct)

    def createToolBar(self):
        # Store the toolbar so extensions can add actions.
        self.toolBar = QToolBar("Tools", self)
        self.addToolBar(self.toolBar)
        self.toolBar.addAction(self.openAct)

    def openImage(self):
        default_folder = self.config.get("default_image_folder", "")
        fileName, _ = QFileDialog.getOpenFileName(
            self,
            "Open Image",
            default_folder,
            "Image Files (*.png *.jpg *.jpeg);;All Files (*)"
        )
        if fileName:
            pixmap = QPixmap(fileName)
            self.image_view.setImage(pixmap)

    def checkDevMode(self):
        # If DEV mode is enabled, automatically load the default image.
        if self.config.get("DEV", False):
            default_image = self.config.get("default_image", "")
            if default_image and os.path.exists(default_image):
                pixmap = QPixmap(default_image)
                self.image_view.setImage(pixmap)
            else:
                print("Default image file not found:", default_image)
