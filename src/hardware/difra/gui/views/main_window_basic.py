import json
import os
from pathlib import Path

from PyQt5.QtGui import QIcon, QPixmap
from PyQt5.QtWidgets import (
    QAction,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QToolBar,
    QVBoxLayout,
)

from hardware.difra.hardware.camera_capture_dialog import CameraCaptureDialog
from hardware.difra.utils.logger import get_module_logger

logger = get_module_logger(__name__)


class MainWindowBasic(QMainWindow):
    def __init__(self, parent=None):
        super().__init__(parent)
        base_title = "EosDX Scanning Software"
        self.setWindowTitle(base_title)

        # Window icon - use platform-specific formats for best display
        import sys
        logo_dir = Path(__file__).resolve().parent.parent.parent / "resources/images"
        if sys.platform == 'win32':
            # Windows: use .ico format
            logo_path = logo_dir / "rick_final.ico"
            if not logo_path.exists():
                logo_path = logo_dir / "rick_final.png"  # Fallback to PNG
        elif sys.platform == 'darwin':
            # macOS: use .icns format for proper dock icon display
            logo_path = logo_dir / "rick_final.icns"
            if not logo_path.exists():
                logo_path = logo_dir / "rick_final.png"  # Fallback to PNG
        else:
            # Linux: use .png format
            logo_path = logo_dir / "rick_final.png"
        
        if logo_path.exists():
            self.setWindowIcon(QIcon(str(logo_path)))
            logger.debug("Loaded window icon", path=str(logo_path))
        else:
            logger.warning("Logo file not found", path=str(logo_path), checked_dir=str(logo_dir))

        self.resize(800, 600)

        # Load config (global + selected setup) and remember base paths
        self._config_dir = (
            Path(__file__).resolve().parent.parent.parent / "resources/config"
        )
        self._global_path = self._config_dir / "global.json"
        self._setups_dir = self._config_dir / "setups"
        main_name = "main_win.json" if os.name == "nt" else "main.json"
        legacy_candidate = self._config_dir / main_name
        self._legacy_main_path = (
            legacy_candidate
            if legacy_candidate.exists()
            else self._config_dir / "main.json"
        )
        self.config = self.load_config()

        # Central image view
        # self.image_view = ImageView(self)
        # self.setCentralWidget(self.image_view)

        # Actions, menus, toolbar
        self.create_actions()
        self.create_menus()
        self.create_tool_bar()

        # Reflect DEV mode visually
        self.update_dev_visuals()

    def load_config(self):
        """Load global config and merge with a selected setup config.
        Fallback to legacy main.json if split configs are missing.
        """
        folder_keys = (
            "difra_base_folder",
            "technical_folder",
            "technical_archive_folder",
            "measurements_folder",
            "measurements_archive_folder",
        )

        def _read_json(p: Path):
            try:
                return json.loads(p.read_text()) if p.exists() else {}
            except Exception as e:
                logger.error("Error reading JSON", error=str(e), path=str(p))
                return {}

        # Legacy fallback
        if not self._global_path.exists() or not self._setups_dir.exists():
            cfg = _read_json(self._legacy_main_path)
            if not cfg:
                logger.error("No configuration found.")
            # Remember legacy path for the in-app editor
            self._active_config_path = self._legacy_main_path
            return cfg

        global_cfg = _read_json(self._global_path)
        # Determine setup: CLI arg --setup, QSettings last selection, or default from global
        setup_name = None
        try:
            import argparse

            parser = argparse.ArgumentParser(add_help=False)
            parser.add_argument("--setup", dest="setup", default=None)
            args, _unknown = parser.parse_known_args()
            setup_name = args.setup
        except Exception:
            setup_name = None

        from PyQt5.QtCore import QSettings

        settings = QSettings("EOSDx", "DiFRA")
        if not setup_name:
            setup_name = settings.value("lastSetup", type=str)
        if not setup_name:
            setup_name = global_cfg.get("default_setup")

        # If still not chosen or file missing, prompt user
        setup_path = None
        if setup_name:
            candidate = (self._setups_dir / f"{setup_name}.json").resolve()
            if candidate.exists():
                setup_path = candidate
        if setup_path is None:
            setup_name, setup_path = self._prompt_for_setup()

        setup_cfg = _read_json(setup_path) if setup_path else {}

        # Persist chosen setup
        if setup_name:
            settings.setValue("lastSetup", setup_name)

        # Merge: setup overrides global where keys overlap
        merged = dict(global_cfg)
        merged.update(setup_cfg)

        # On Windows, prefer folder paths from main_win.json when available.
        if os.name == "nt":
            legacy_cfg = _read_json(self._legacy_main_path)
            for key in folder_keys:
                value = legacy_cfg.get(key)
                if value:
                    merged[key] = value

        # Remember active config path for editor
        self._active_config_path = setup_path if setup_path else self._legacy_main_path
        return merged

    def create_actions(self):
        # File open
        self.open_act = QAction("Open Image", self, triggered=self.open_image)
        # Camera
        self.capture_camera_act = QAction(
            "Capture from Camera", self, triggered=self.capture_from_camera
        )
        # Edit config dialog
        self.edit_config_act = QAction("Edit Setup Config…", self, triggered=self.edit_config)
        # Edit global settings
        self.edit_global_act = QAction("Edit Global Settings…", self, triggered=self.edit_global_settings)
        # Toggle DEV/demo mode
        self.toggle_dev_act = QAction("", self, triggered=self.toggle_dev_mode)
        # Help - README
        self.readme_act = QAction("Documentation (README)", self, triggered=self.open_readme)

    def create_menus(self):
        file_menu = self.menuBar().addMenu("File")
        file_menu.addAction(self.open_act)
        file_menu.addAction(self.capture_camera_act)
        settings_menu = self.menuBar().addMenu("Settings")
        settings_menu.addAction(self.edit_config_act)
        settings_menu.addAction(self.edit_global_act)
        settings_menu.addSeparator()
        settings_menu.addAction(self.readme_act)

    def create_tool_bar(self):
        self.toolbar = QToolBar("Tools", self)
        self.toolbar.setObjectName("MainToolsToolbar")
        self.addToolBar(self.toolbar)
        self.toolbar.addAction(self.open_act)
        self.toolbar.addAction(self.capture_camera_act)
        self.toolbar.addSeparator()
        self.toolbar.addAction(self.toggle_dev_act)

    def open_image(self):
        default_folder = self.config.get("default_image_folder", "")
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Open Image",
            default_folder,
            "Image Files (*.png *.jpg *.jpeg);;All Files (*)",
        )
        if path:
            pixmap = QPixmap(path)
            self.image_view.set_image(pixmap, image_path=path)
            try:
                self.delete_all_shapes_from_table()
                self.delete_all_points()
            except Exception as e:
                logger.warning(
                    "Error clearing shapes/points after image load", error=str(e)
                )
            
            # Auto-create session when new sample image is loaded
            if hasattr(self, '_handle_new_sample_image'):
                self._handle_new_sample_image(path)

    def capture_from_camera(self):
        default_folder = self.config.get("default_folder", "")
        dialog = CameraCaptureDialog(self, default_folder=default_folder)
        if dialog.exec_() == QDialog.Accepted:
            image_path = getattr(dialog, "selected_image_path", None)
            if image_path and os.path.exists(image_path):
                pixmap = QPixmap(image_path)
                self.image_view.set_image(pixmap, image_path=image_path)
                try:
                    self.delete_all_shapes_from_table()
                    self.delete_all_points()
                except Exception as e:
                    logger.warning(
                        "Error clearing shapes/points after camera capture",
                        error=str(e),
                    )
                
                # Auto-create session when new sample image is captured
                if hasattr(self, '_handle_new_sample_image'):
                    self._handle_new_sample_image(image_path)
            else:
                QMessageBox.warning(
                    self,
                    "Load Error",
                    "Image was not saved or cannot be found. "
                    "Please check the folder and try again.",
                )

    def check_dev_mode(self):
        if self.config.get("DEV", False):
            default_image = self.config.get("default_image", "")
            if default_image and os.path.exists(default_image):
                pixmap = QPixmap(default_image)
                self.image_view.set_image(pixmap, image_path=default_image)
            else:
                logger.warning(
                    "Default image file not found in DEV mode", path=default_image
                )

    def edit_config(self):
        """
        Open a JSON editor for the currently active setup config file (setup or legacy), save and reload config.
        """
        target_path = getattr(self, "_active_config_path", self._legacy_main_path)
        try:
            text = target_path.read_text()
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Cannot open config:\n{e}")
            return

        dlg = QDialog(self)
        dlg.setWindowTitle(f"Edit Setup Config - {target_path.name}")
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
                target_path.write_text(json.dumps(parsed, indent=4))
            except Exception as write_e:
                QMessageBox.critical(self, "Error", f"Cannot write config:\n{write_e}")
                return
            # Recompute full config if we edited a setup file; else simple assign
            if (
                target_path.name.endswith(".json")
                and target_path.parent.name == "setups"
            ):
                # Reload merged config (global + setup)
                self.config = self.load_config()
            else:
                self.config = parsed
            self.update_dev_visuals()
            QMessageBox.information(self, "Config Saved", "Configuration reloaded.")
            dlg.accept()

        buttons.accepted.connect(on_save)
        buttons.rejected.connect(dlg.reject)

        dlg.resize(600, 400)
        dlg.exec_()

    def edit_global_settings(self):
        """
        Open a JSON editor for the global.json file, save and reload config.
        """
        target_path = self._global_path
        try:
            text = target_path.read_text()
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Cannot open global settings:\n{e}")
            return

        dlg = QDialog(self)
        dlg.setWindowTitle("Edit Global Settings - global.json")
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
                target_path.write_text(json.dumps(parsed, indent=4))
            except Exception as write_e:
                QMessageBox.critical(self, "Error", f"Cannot write global settings:\n{write_e}")
                return
            # Reload merged config (global + current setup)
            self.config = self.load_config()
            self.update_dev_visuals()
            QMessageBox.information(self, "Settings Saved", "Global settings reloaded.")
            dlg.accept()

        buttons.accepted.connect(on_save)
        buttons.rejected.connect(dlg.reject)

        dlg.resize(600, 400)
        dlg.exec_()

    def open_readme(self):
        """Open README.md file with system default application."""
        readme_path = (
            Path(__file__).resolve().parent.parent.parent / "README.md"
        )
        if not readme_path.exists():
            QMessageBox.warning(
                self,
                "README Not Found",
                f"README file not found at:\n{readme_path}"
            )
            return
        
        try:
            import subprocess
            import sys
            
            if sys.platform == 'darwin':  # macOS
                subprocess.Popen(['open', str(readme_path)])
            elif sys.platform == 'win32':  # Windows
                subprocess.Popen(['start', str(readme_path)], shell=True)
            else:  # Linux
                subprocess.Popen(['xdg-open', str(readme_path)])
            logger.info("Opened README", path=str(readme_path))
        except Exception as e:
            QMessageBox.warning(
                self,
                "Error Opening README",
                f"Could not open README file:\n{e}\n\nPath: {readme_path}"
            )
            logger.error("Failed to open README", error=str(e), path=str(readme_path))

    def _prompt_for_setup(self):
        """Prompt user to choose a setup JSON from the setups directory.
        Returns (setup_name, setup_path).
        """
        try:
            from PyQt5.QtWidgets import (
                QDialog,
                QLabel,
                QListWidget,
                QPushButton,
                QVBoxLayout,
            )
        except Exception:
            return None, None

        dlg = QDialog(self)
        dlg.setWindowTitle("Select Experimental Setup")
        layout = QVBoxLayout(dlg)
        layout.addWidget(QLabel("Choose an experimental setup:"))
        lst = QListWidget(dlg)
        setup_files = sorted([p for p in self._setups_dir.glob("*.json")])
        for p in setup_files:
            lst.addItem(p.stem)
        if not setup_files:
            layout.addWidget(QLabel("No setups found under resources/config/setups/"))
        layout.addWidget(lst)
        ok = QPushButton("OK", dlg)
        ok.clicked.connect(dlg.accept)
        layout.addWidget(ok)
        dlg.resize(380, 300)
        if dlg.exec_() == QDialog.Accepted and lst.currentItem():
            name = lst.currentItem().text()
            path = (self._setups_dir / f"{name}.json").resolve()
            return name, path if path.exists() else (name, None)
        return None, None

    def update_dev_visuals(self):
        """
        Gray background + "[DEMO]" when DEV=True, else normal.
        """
        base_title = "EosDX Scanning Software"
        is_dev = self.config.get("DEV", False)
        if is_dev:
            self.setStyleSheet("background-color: lightgray;")
            self.setWindowTitle(f"{base_title} [DEMO]")
            self.toggle_dev_act.setText("Switch to Production")
        else:
            self.setStyleSheet("")
            self.setWindowTitle(base_title)
            self.toggle_dev_act.setText("Switch to Demo")

    def toggle_dev_mode(self):
        """
        Flip DEV flag in the global config if present (fallback to active file).
        """
        new_dev = not self.config.get("DEV", False)
        self.config["DEV"] = new_dev
        target = (
            self._global_path
            if self._global_path.exists()
            else getattr(self, "_active_config_path", self._legacy_main_path)
        )
        try:
            data = json.loads(target.read_text()) if target.exists() else {}
            data["DEV"] = new_dev
            target.write_text(json.dumps(data, indent=4))
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Cannot write config file:\n{e}")
            return
        self.update_dev_visuals()
        if hasattr(self, "on_config_mode_changed"):
            try:
                self.on_config_mode_changed(new_dev)
            except Exception as exc:
                logger.warning("Failed to apply mode switch updates", error=str(exc))
