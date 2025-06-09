import os
import json
from pathlib import Path

from PyQt5.QtWidgets import (
    QMainWindow, QAction, QFileDialog, QToolBar,
    QDialog, QPlainTextEdit, QDialogButtonBox, QVBoxLayout, QMessageBox
)
from PyQt5.QtGui import QPixmap, QIcon

from hardware.Ulster.gui.views.image_view import ImageView


class MainWindowBasic(QMainWindow):
    def __init__(self, parent=None):
        super().__init__(parent)
        base_title = "EosDX Scanning Software"
        self.setWindowTitle(base_title)

        # Window icon
        logo_path = Path(__file__).resolve().parent.parent.parent / 'resources/images/rick_final.png'
        if logo_path.exists():
            self.setWindowIcon(QIcon(str(logo_path)))
        else:
            print("Logo file not found:", logo_path)

        self.resize(800, 600)

        # Load config and remember path
        config_path = Path(__file__).resolve().parent.parent.parent / 'resources/config/main.json'
        self._config_path = config_path
        self.config = self.load_config()

        # Central image view
        self.image_view = ImageView(self)
        self.setCentralWidget(self.image_view)

        # Actions, menus, toolbar
        self.create_actions()
        self.create_menus()
        self.create_tool_bar()

        # Reflect DEV mode visually
        self.update_dev_visuals()

        # If DEV mode, auto-open default image
        self.check_dev_mode()

    def load_config(self):
        try:
            with open(self._config_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            print("Error loading config:", e)
            return {}

    def create_actions(self):
        # File open
        self.open_act = QAction("Open Image", self, triggered=self.open_image)
        # Edit config dialog
        self.editConfigAct = QAction("Edit Config…", self, triggered=self.edit_config)
        # Toggle DEV/demo mode
        self.toggleDevAct  = QAction("", self, triggered=self.toggle_dev_mode)

    def create_menus(self):
        fileMenu = self.menuBar().addMenu("File")
        fileMenu.addAction(self.open_act)

        settingsMenu = self.menuBar().addMenu("Settings")
        settingsMenu.addAction(self.editConfigAct)

    def create_tool_bar(self):
        self.toolbar = QToolBar("Tools", self)
        self.addToolBar(self.toolbar)
        self.toolbar.addAction(self.open_act)
        self.toolbar.addSeparator()
        self.toolbar.addAction(self.toggleDevAct)

    def open_image(self):
        default_folder = self.config.get("default_image_folder", "")
        path, _ = QFileDialog.getOpenFileName(
            self, "Open Image", default_folder,
            "Image Files (*.png *.jpg *.jpeg);;All Files (*)"
        )
        if path:
            pixmap = QPixmap(path)
            self.image_view.set_image(pixmap, image_path=path)
            try:
                self.delete_all_shapes_from_table()
                self.delete_all_points()
            except Exception:
                pass

    def check_dev_mode(self):
        if self.config.get("DEV", False):
            default_image = self.config.get("default_image", "")
            if default_image and os.path.exists(default_image):
                pixmap = QPixmap(default_image)
                self.image_view.set_image(pixmap, image_path=default_image)
            else:
                print("Default image file not found:", default_image)

    def edit_config(self):
        """
        Pop up a JSON editor for main.json; save and reload config.
        """
        try:
            text = self._config_path.read_text()
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Cannot open config:\n{e}")
            return

        dlg = QDialog(self)
        dlg.setWindowTitle("Edit Config")
        layout = QVBoxLayout(dlg)

        editor = QPlainTextEdit(dlg)
        editor.setPlainText(text)
        layout.addWidget(editor)

        buttons = QDialogButtonBox(QDialogButtonBox.Save | QDialogButtonBox.Cancel, dlg)
        layout.addWidget(buttons)

        def on_save():
            new_text = editor.toPlainText()
            try:
                parsed = json.loads(new_text)
            except Exception as parse_e:
                QMessageBox.warning(dlg, "JSON Error", f"Invalid JSON:\n{parse_e}")
                return
            try:
                self._config_path.write_text(json.dumps(parsed, indent=4))
            except Exception as write_e:
                QMessageBox.critical(self, "Error", f"Cannot write config:\n{write_e}")
                return
            self.config = parsed
            self.update_dev_visuals()
            QMessageBox.information(self, "Config Saved", "Configuration reloaded.")
            dlg.accept()

        buttons.accepted.connect(on_save)
        buttons.rejected.connect(dlg.reject)

        dlg.resize(600, 400)
        dlg.exec_()

    def update_dev_visuals(self):
        """
        Gray background + "[DEMO]" when DEV=True, else normal.
        """
        base_title = "EosDX Scanning Software"
        is_dev = self.config.get("DEV", False)
        if is_dev:
            self.setStyleSheet("background-color: lightgray;")
            self.setWindowTitle(f"{base_title} [DEMO]")
            self.toggleDevAct.setText("Switch to Production")
        else:
            self.setStyleSheet("")
            self.setWindowTitle(base_title)
            self.toggleDevAct.setText("Switch to Demo")

    def toggle_dev_mode(self):
        """
        Flip DEV flag, persist, and update visuals.
        """
        new_dev = not self.config.get("DEV", False)
        self.config["DEV"] = new_dev
        try:
            with open(self._config_path, 'w') as f:
                json.dump(self.config, f, indent=4)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Could not save config:\n{e}")
            return
        self.update_dev_visuals()
