import logging
import random
import sys
from pathlib import Path

# Set the project root.
project_root = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from PyQt5.QtCore import QDate, QSettings
from PyQt5.QtWidgets import QApplication, QMessageBox

from hardware.EosDxDc.gui.views.main_window import MainWindow
from hardware.EosDxDc.resources.motivation import motivation_phrases
from hardware.EosDxDc.utils.logging_setup import (
    configure_third_party_logging,
    log_context,
    setup_logging,
)

# Setup enhanced logging
log_config = {
    "console_level": logging.INFO,
    "file_level": logging.DEBUG,
    "max_bytes": 20 * 1024 * 1024,  # 20MB
    "backup_count": 10,
}
log_path = setup_logging(config=log_config, structured=True)
configure_third_party_logging()

logger = logging.getLogger(__name__)


def show_motivation_dialog(parent=None):
    msg = QMessageBox(parent)
    msg.setWindowTitle("Welcome!")
    phrase = random.choice(motivation_phrases)
    msg.setText(phrase)
    msg.setIcon(QMessageBox.Information)
    msg.setStandardButtons(QMessageBox.Ok)
    msg.exec_()  # blocks until OK is clicked


if __name__ == "__main__":
    with log_context(
        session_id=f"session_{QDate.currentDate().toString('yyyy-MM-dd')}",
        hardware_state="initializing",
    ):
        logger.info("EOSDxDc application starting", extra={"log_path": str(log_path)})

        app = QApplication(sys.argv)

        # --- only show once per day ---
        settings = QSettings("EOSDx", "EOSDxDc")
        last_date_shown = settings.value("lastMotivationDate", "")
        today = QDate.currentDate().toString("yyyy-MM-dd")

        if last_date_shown != today:
            logger.debug("Showing motivation dialog")
            show_motivation_dialog()
            settings.setValue("lastMotivationDate", today)
        else:
            logger.debug("Skipping motivation dialog (already shown today)")
        # --------------------------------

        logger.info("Creating main window")
        win = MainWindow()
        win.setWindowTitle("EOSDxDc")
        win.show()

        logger.info("EOSDxDc application ready")
        exit_code = app.exec_()
        logger.info("EOSDxDc application shutting down", extra={"exit_code": exit_code})
        sys.exit(exit_code)
