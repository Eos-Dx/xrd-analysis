import base64
import json
import re
from pathlib import Path

from PyQt5.QtCore import QSettings, Qt
from PyQt5.QtWidgets import (
    QCheckBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
)


def _parse_distance_from_poni(poni_text: str):
    if not poni_text:
        return None
    m = re.search(r"^Distance:\s*([0-9.eE+-]+)", poni_text, flags=re.MULTILINE)
    try:
        return float(m.group(1)) if m else None
    except Exception:
        return None


class PreflightDialog(QDialog):
    """Mandatory pre-capture checklist with confirmations.

    Continue is disabled until all checks pass AND all checkboxes are ticked.
    Can be globally disabled with an admin password (stored in QSettings).
    """

    def __init__(
        self,
        parent,
        measurement_folder: Path,
        state_file: Path,
        ponis: dict,
        attenuation_on: bool,
    ):
        super().__init__(parent)
        self.setWindowTitle("Preflight Checks")
        self.setModal(True)
        self.setMinimumWidth(540)
        self.measurement_folder = Path(measurement_folder or "")
        self.state_file = Path(state_file or "state.json")
        self.ponis = ponis or {}
        self.attenuation_on = bool(attenuation_on)

        self._setup_ui()
        self._revalidate()

    def _setup_ui(self):
        layout = QVBoxLayout(self)

        # State path and distance
        self.state_label = QLabel("")
        self.state_confirm = QCheckBox(
            "I confirm the state file will be saved in the correct folder and at the correct distance."
        )
        layout.addWidget(self.state_label)
        layout.addWidget(self.state_confirm)

        # Meta file
        row = QHBoxLayout()
        self.meta_label = QLabel("")
        self.meta_browse_btn = QPushButton("Browse…")
        self.meta_browse_btn.clicked.connect(self._browse_meta)
        row.addWidget(self.meta_label, 1)
        row.addWidget(self.meta_browse_btn)
        layout.addLayout(row)
        self.meta_confirm = QCheckBox(
            "I confirm a metadata file (technical_meta_*.json) is present and correctly named."
        )
        layout.addWidget(self.meta_confirm)

        # Attenuation
        self.atten_label = QLabel("")
        layout.addWidget(self.atten_label)
        self.atten_confirm = QCheckBox(
            "I confirm the attenuation setting is correct (ON/OFF) for this measurement."
        )
        layout.addWidget(self.atten_confirm)

        # Admin unlock
        admin_row = QHBoxLayout()
        self.admin_btn = QPushButton("Admin unlock…")
        self.admin_btn.clicked.connect(self._admin_unlock)
        self.unlock_status = QLabel("")
        self.unlock_status.setStyleSheet("color: #666;")
        admin_row.addWidget(self.admin_btn)
        admin_row.addWidget(self.unlock_status, 1)
        layout.addLayout(admin_row)

        # Buttons
        self.buttons = QDialogButtonBox(QDialogButtonBox.Cancel | QDialogButtonBox.Ok)
        self.buttons.button(QDialogButtonBox.Ok).setText("Continue")
        self.buttons.button(QDialogButtonBox.Ok).setEnabled(False)
        self.buttons.rejected.connect(self.reject)
        self.buttons.accepted.connect(self._accept_if_valid)
        layout.addWidget(self.buttons)

        # Wire checkboxes to revalidate
        self.state_confirm.toggled.connect(self._revalidate)
        self.meta_confirm.toggled.connect(self._revalidate)
        self.atten_confirm.toggled.connect(self._revalidate)

    # --- Validation helpers ---
    def _current_distance(self):
        # Use the first available PONI distance
        for txt in (self.ponis or {}).values():
            d = _parse_distance_from_poni(txt or "")
            if d is not None:
                return d
        return None

    def _expected_state_path(self):
        return self.measurement_folder / self.state_file.name

    def _detect_meta(self):
        try:
            for p in self.measurement_folder.glob("technical_meta_*.json"):
                return p
        except Exception:
            pass
        return None

    def _revalidate(self):
        # State + distance info
        dist = self._current_distance()
        state_expected = self._expected_state_path()
        ok_folder = state_expected.parent.resolve() == self.measurement_folder.resolve()
        state_txt = f"State: {state_expected} | Distance: {dist if dist is not None else 'unknown'}"
        icon = "✅" if ok_folder else "⚠️"
        self.state_label.setText(f"{icon} {state_txt}")

        # Meta
        meta = self._detect_meta()
        icon_meta = "✅" if meta else "⚠️"
        self.meta_label.setText(
            f"{icon_meta} Meta: {str(meta) if meta else 'technical_meta_*.json not found'}"
        )

        # Attenuation
        icon_att = "✅" if True else "⚠️"
        self.atten_label.setText(
            f"{icon_att} Attenuation: {'ON' if self.attenuation_on else 'OFF'}"
        )

        # Enable Continue only if all confirms checked and basic validations are OK
        all_confirmed = (
            self.state_confirm.isChecked()
            and self.meta_confirm.isChecked()
            and self.atten_confirm.isChecked()
        )
        ok_basic = (
            ok_folder and True
        )  # meta presence not strictly mandatory, but confirmation is
        enable = all_confirmed and ok_basic and not self._is_preflight_disabled()
        self.buttons.button(QDialogButtonBox.Ok).setEnabled(enable)

    def _browse_meta(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select metadata file", str(self.measurement_folder), "JSON (*.json)"
        )
        if path:
            # Update label; validation relies on presence/confirmation
            self.meta_label.setText(f"✅ Meta: {path}")
            # auto-check confirm to save clicks? leave to user.
        self._revalidate()

    # --- Admin unlock ---
    def _is_preflight_disabled(self):
        s = QSettings("EOSDx", "EOSDxDc")
        return bool(s.value("preflight_disabled", False, type=bool))

    def _admin_unlock(self):
        s = QSettings("EOSDx", "EOSDxDc")
        stored_hash = s.value("preflight_admin_password_hash", "", type=str)

        # Simple inline prompt
        dlg = QDialog(self)
        dlg.setWindowTitle("Admin Unlock")
        v = QVBoxLayout(dlg)
        v.addWidget(QLabel("Enter admin password (or set a new one if none exists):"))
        pwd = QLineEdit()
        pwd.setEchoMode(QLineEdit.Password)
        v.addWidget(pwd)
        bb = QDialogButtonBox(QDialogButtonBox.Cancel | QDialogButtonBox.Ok)
        v.addWidget(bb)
        bb.rejected.connect(dlg.reject)
        bb.accepted.connect(dlg.accept)
        if dlg.exec_() != QDialog.Accepted:
            return

        import hashlib

        entered = pwd.text().strip()
        h = hashlib.sha256(("EOSDX:" + entered).encode("utf-8")).hexdigest()
        if not stored_hash:
            # Set new password
            s.setValue("preflight_admin_password_hash", h)
            s.sync()
            QMessageBox.information(self, "Admin", "Admin password set.")
            self.unlock_status.setText("Admin password set.")
        elif h != stored_hash:
            QMessageBox.warning(self, "Admin", "Invalid password.")
            return

        # Disable preflight globally
        s.setValue("preflight_disabled", True)
        s.sync()
        self.unlock_status.setText("Preflight disabled (admin)")
        # Also allow proceeding now
        self.accept()

    def _accept_if_valid(self):
        if self.buttons.button(QDialogButtonBox.Ok).isEnabled():
            self.accept()
        else:
            QMessageBox.warning(self, "Preflight", "Please complete all confirmations.")
