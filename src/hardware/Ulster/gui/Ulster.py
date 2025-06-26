import random
import sys
from pathlib import Path

# Set the project root.
project_root = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from PyQt5.QtWidgets import QApplication, QMessageBox

from hardware.Ulster.gui.views.main_window import MainWindow
from hardware.Ulster.resources.motivation import motivation_phrases


def show_motivation_dialog(parent=None):
    msg = QMessageBox(parent)
    msg.setWindowTitle("Welcome!")
    # pick a random phrase
    phrase = random.choice(motivation_phrases)
    msg.setText(phrase)
    msg.setIcon(QMessageBox.Information)
    msg.setStandardButtons(QMessageBox.Ok)
    msg.exec_()  # blocks until OK is clicked


if __name__ == "__main__":
    app = QApplication(sys.argv)

    # show a random motivational phrase first
    show_motivation_dialog()

    # then proceed as normal
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())
