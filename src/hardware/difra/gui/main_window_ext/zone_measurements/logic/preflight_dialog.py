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
        self.technical_h5_path = None

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

        # Technical HDF5 container
        row_h5 = QHBoxLayout()
        self.h5_label = QLabel("")
        self.h5_browse_btn = QPushButton("Browse…")
        self.h5_browse_btn.clicked.connect(self._browse_h5)
        row_h5.addWidget(self.h5_label, 1)
        row_h5.addWidget(self.h5_browse_btn)
        layout.addLayout(row_h5)
        self.h5_confirm = QCheckBox(
            "I confirm the technical HDF5 container (technical_*.h5) exists and is valid."
        )
        layout.addWidget(self.h5_confirm)

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

        # Buttons
        self.buttons = QDialogButtonBox(QDialogButtonBox.Cancel | QDialogButtonBox.Ok)
        self.buttons.button(QDialogButtonBox.Ok).setText("Continue")
        self.buttons.button(QDialogButtonBox.Ok).setEnabled(False)
        self.buttons.rejected.connect(self.reject)
        self.buttons.accepted.connect(self._accept_if_valid)
        layout.addWidget(self.buttons)

        # Wire checkboxes to revalidate
        self.state_confirm.toggled.connect(self._revalidate)
        self.h5_confirm.toggled.connect(self._revalidate)
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

    def _detect_h5(self):
        """Detect technical HDF5 container in measurement folder."""
        try:
            for p in self.measurement_folder.glob("technical_*.h5"):
                self.technical_h5_path = p
                return p
        except Exception:
            pass
        self.technical_h5_path = None
        return None

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

        # Technical HDF5 container
        h5 = self._detect_h5()
        icon_h5 = "✅" if h5 else "⚠️"
        self.h5_label.setText(
            f"{icon_h5} Technical H5: {str(h5.name) if h5 else 'technical_*.h5 not found'}"
        )

        # Meta
        meta = self._detect_meta()
        icon_meta = "✅" if meta else "⚠️"
        self.meta_label.setText(
            f"{icon_meta} Meta: {str(meta.name) if meta else 'technical_meta_*.json not found'}"
        )

        # Attenuation
        icon_att = "✅" if True else "⚠️"
        self.atten_label.setText(
            f"{icon_att} Attenuation: {'ON' if self.attenuation_on else 'OFF'}"
        )

        # Enable Continue only if all confirms checked and basic validations are OK
        all_confirmed = (
            self.state_confirm.isChecked()
            and self.h5_confirm.isChecked()
            and self.meta_confirm.isChecked()
            and self.atten_confirm.isChecked()
        )
        ok_basic = (
            ok_folder and h5 is not None
        )  # H5 container is required, meta/confirmations are mandatory
        enable = all_confirmed and ok_basic
        self.buttons.button(QDialogButtonBox.Ok).setEnabled(enable)

    def _browse_h5(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select technical HDF5 container",
            str(self.measurement_folder),
            "HDF5 Files (*.h5 *.hdf5);;All Files (*)",
        )
        if path:
            self.technical_h5_path = Path(path)
            self.h5_label.setText(f"✅ Technical H5: {Path(path).name}")
        self._revalidate()

    def _browse_meta(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select metadata file", str(self.measurement_folder), "JSON (*.json)"
        )
        if path:
            # Update label; validation relies on presence/confirmation
            self.meta_label.setText(f"✅ Meta: {Path(path).name}")
            # auto-check confirm to save clicks? leave to user.
        self._revalidate()

    # --- Admin unlock ---

    def _accept_if_valid(self):
        if self.buttons.button(QDialogButtonBox.Ok).isEnabled():
            self.accept()
        else:
            QMessageBox.warning(self, "Preflight", "Please complete all confirmations.")
