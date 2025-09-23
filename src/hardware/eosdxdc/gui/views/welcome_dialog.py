import random
from pathlib import Path

from PyQt5.QtCore import QSettings, Qt
from PyQt5.QtGui import QIcon, QPixmap
from PyQt5.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QLabel,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from hardware.eosdxdc.resources.motivation import motivation_phrases


class WelcomeDialog(QDialog):
    """Startup welcome dialog shown on every start.

    - Displays a stylized header with the app logo (D^2_xc) and title.
    - Lists each setup as a big colorful button.
    - Shows a motivation phrase at the bottom.
    - Saves the chosen setup to QSettings("EOSDx", "EOSDxDc") as "lastSetup".
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Welcome to EosDxDc")
        self.setModal(True)
        self.resize(560, 460)

        # Resolve setups directory and defaults
        app_root = Path(__file__).resolve().parent.parent.parent
        config_dir = app_root / "resources" / "config"
        self.setups_dir = config_dir / "setups"
        global_path = config_dir / "global.json"

        # Set dialog icon to match main app
        logo_path = app_root / "resources/images/rick_final.png"
        if logo_path.exists():
            self.setWindowIcon(QIcon(str(logo_path)))

        # QSettings
        self.settings = QSettings("EOSDx", "EOSDxDc")
        last_setup = self.settings.value("lastSetup", type=str)
        default_setup = None
        try:
            if global_path.exists():
                import json

                g = json.loads(global_path.read_text())
                default_setup = g.get("default_setup")
        except Exception:
            default_setup = None

        # Collect setup names
        self._setup_names = [p.stem for p in sorted(self.setups_dir.glob("*.json"))]
        # Preferred preselection (not strictly needed since each setup is a button)
        self._preferred = last_setup or default_setup

        # --- Dialog-wide colorful style ---
        self.setStyleSheet(
            """
            QDialog {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                    stop:0 #f0f7ff, stop:1 #e6fffa);
            }
            QLabel#TitleLabel {
                color: #0d47a1;
                font-size: 22px;
                font-weight: 700;
            }
            QLabel#LogoLabel {
                color: #2e7d32;
                font-size: 40px;
                font-weight: 900;
            }
            QLabel#MotivationLabel {
                color: #004d40;
                font-size: 12px;
                font-style: italic;
                padding: 8px;
                border-radius: 6px;
                background: rgba(255,255,255,0.5);
            }
            QPushButton.SetupButton {
                background-color: #1976d2;
                color: white;
                border: 0px solid #1565c0;
                border-radius: 8px;
                padding: 10px 14px;
                font-size: 14px;
                font-weight: 600;
            }
            QPushButton.SetupButton:hover {
                background-color: #1e88e5;
            }
            QPushButton.SetupButton:pressed {
                background-color: #1565c0;
            }
            QScrollArea {
                border: none;
                background: transparent;
            }
            """
        )

        # Build UI
        layout = QVBoxLayout(self)

        # Header area with title and logo
        header = QWidget(self)
        header_layout = QVBoxLayout(header)
        header_layout.setContentsMargins(6, 6, 6, 12)
        title = QLabel("Welcome to EosDxDc", header)
        title.setObjectName("TitleLabel")
        title.setAlignment(Qt.AlignHCenter)

        # Use same image as main app for the welcome dialog header
        logo_label = QLabel(header)
        logo_label.setObjectName("LogoLabel")
        logo_label.setAlignment(Qt.AlignHCenter)
        if logo_path.exists():
            pm = QPixmap(str(logo_path))
            # Scale to a reasonable height while keeping aspect ratio
            scaled = pm.scaledToHeight(96, mode=Qt.SmoothTransformation)
            logo_label.setPixmap(scaled)
        else:
            # Fallback text if image is missing
            logo_label.setText("D^2_xc")

        header_layout.addWidget(title)
        header_layout.addWidget(logo_label)
        layout.addWidget(header)

        # Prompt
        prompt = QLabel("Choose an experimental setup for this session:", self)
        prompt.setAlignment(Qt.AlignHCenter)
        layout.addWidget(prompt)

        # Scroll area with setup buttons
        scroll = QScrollArea(self)
        scroll.setWidgetResizable(True)
        inner = QWidget(scroll)
        inner_layout = QVBoxLayout(inner)
        inner_layout.setSpacing(8)

        if not self._setup_names:
            inner_layout.addWidget(
                QLabel("No setups found under resources/config/setups/", inner)
            )
        else:
            for name in self._setup_names:
                btn = QPushButton(name, inner)
                btn.setProperty("class", "SetupButton")
                btn.setObjectName(f"btn_{name}")
                btn.setCursor(Qt.PointingHandCursor)
                btn.setMinimumHeight(36)
                btn.setStyleSheet("QPushButton { margin: 2px 8px; }")
                # Highlight preferred default
                if self._preferred and name == self._preferred:
                    btn.setText(f"{name} (default)")
                btn.setProperty("cssClass", "SetupButton")
                btn.setAccessibleName("SetupButton")
                btn.setAccessibleDescription(f"Select setup {name}")
                btn.clicked.connect(lambda _c=False, n=name: self._choose_setup(n))
                inner_layout.addWidget(btn)

        inner_layout.addStretch(1)
        inner.setLayout(inner_layout)
        scroll.setWidget(inner)
        layout.addWidget(scroll, 1)

        # Motivation text at bottom
        mot = random.choice(motivation_phrases)
        mot_label = QLabel(mot, self)
        mot_label.setObjectName("MotivationLabel")
        mot_label.setWordWrap(True)
        layout.addWidget(mot_label)

        # Cancel button row (optional)
        buttons = QDialogButtonBox(QDialogButtonBox.Cancel, self)
        # Rename Cancel to Quit and make it close the app unless a setup is chosen
        try:
            buttons.button(QDialogButtonBox.Cancel).setText("Quit")
        except Exception:
            pass
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _choose_setup(self, name: str):
        if not name:
            return
        self.settings.setValue("lastSetup", name)
        self.accept()
