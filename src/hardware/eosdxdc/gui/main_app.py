import logging
import sys
from pathlib import Path

# Set the project root.
project_root = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from PyQt5.QtCore import QDate, QSettings
from PyQt5.QtWidgets import QApplication, QMessageBox

from hardware.eosdxdc.gui.views.main_window import MainWindow
from hardware.eosdxdc.gui.views.welcome_dialog import WelcomeDialog
from hardware.eosdxdc.utils.logging_setup import (
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


if __name__ == "__main__":
    with log_context(
        session_id=f"session_{QDate.currentDate().toString('yyyy-MM-dd')}",
        hardware_state="initializing",
    ):
        logger.info("EOSDxDc application starting", extra={"log_path": str(log_path)})

        app = QApplication(sys.argv)

        # --- Welcome dialog with setup selection and embedded motivation ---
        # (Motivation popup removed; now shown inside Welcome dialog)

        # Always show Welcome dialog for setup selection before creating main window
        try:
            logger.debug("Showing welcome dialog for setup selection")
            dlg = WelcomeDialog()
            dlg.exec_()
        except Exception as e:
            logger.warning("Failed to show welcome dialog", exc_info=e)

        logger.info("Creating main window")
        win = MainWindow()
        win.setWindowTitle("EOSDxDc")
        win.show()

        logger.info("EOSDxDc application ready")
        exit_code = app.exec_()
        logger.info("EOSDxDc application shutting down", extra={"exit_code": exit_code})
        sys.exit(exit_code)
