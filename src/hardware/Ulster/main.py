"""Main entry point for Ulster XRD analysis application."""

import sys
from pathlib import Path

# Ensure project root (src) is on sys.path so 'xrdanalysis' and siblings resolve
HERE = Path(__file__).resolve().parent
SRC = HERE.parent.parent  # C:\dev\xrd-analysis\src
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
# Also ensure Ulster package itself is importable when running directly
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from PyQt5.QtWidgets import QApplication

from utils.logging import setup_logging
from config.settings import config_manager
from gui.app import UlsterApp


def main():
    """Main application entry point."""
    
    # Setup logging first
    setup_logging(
        log_level=config_manager.app_settings.log_level,
        log_to_file=config_manager.app_settings.log_to_file,
        dev_mode=config_manager.dev_mode
    )
    
    # Create Qt application
    app = QApplication(sys.argv)
    
    # Create and run Ulster application
    ulster_app = UlsterApp()
    ulster_app.run()
    
    # Start Qt event loop
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
