import random
import sys
from pathlib import Path

from PyQt5.QtCore import QDate, QSettings
from PyQt5.QtWidgets import QApplication, QMessageBox

from .windows.main_window import MainWindow
from ..resources.motivation import motivation_phrases
from utils.logging import setup_logging, log_context, get_module_logger
from config.settings import config_manager

logger = get_module_logger(__name__)


def show_motivation_dialog(parent=None):
    msg = QMessageBox(parent)
    msg.setWindowTitle("Welcome!")
    phrase = random.choice(motivation_phrases)
    msg.setText(phrase)
    msg.setIcon(QMessageBox.Information)
    msg.setStandardButtons(QMessageBox.Ok)
    msg.exec_()  # blocks until OK is clicked


class UlsterApp:
    """Main Ulster application."""
    
    def __init__(self):
        self.main_window = None
        
    def run(self):
        """Initialize and show the main application window."""
        try:
            with log_context(
                session_id=f"session_{QDate.currentDate().toString('yyyy-MM-dd')}",
                hardware_state="initializing",
            ):
                logger.info("Ulster application starting")
                
                # --- Show motivation dialog once per day ---
                settings = QSettings("Ulster", "UlsterApp")
                last_date_shown = settings.value("lastMotivationDate", "")
                today = QDate.currentDate().toString("yyyy-MM-dd")
                
                if last_date_shown != today:
                    logger.debug("Showing motivation dialog")
                    show_motivation_dialog()
                    settings.setValue("lastMotivationDate", today)
                else:
                    logger.debug("Skipping motivation dialog (already shown today)")
                
                # Create main window
                logger.info("Creating main window")
                self.main_window = MainWindow()
                self.main_window.show()
                
                # Check dev mode and load default image if configured
                if config_manager.dev_mode:
                    self.main_window.check_dev_mode()
                
                logger.info("Ulster application ready", 
                           dev_mode=config_manager.dev_mode)
                           
        except Exception as e:
            logger.error("Failed to start Ulster application", error=str(e))
            raise
