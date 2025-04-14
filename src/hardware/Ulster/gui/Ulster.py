from pathlib import Path
import sys

# Set the project root.
project_root = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from PyQt5.QtWidgets import QApplication
from hardware.Ulster.gui.views.main_window import MainWindow


if __name__ == '__main__':
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())
