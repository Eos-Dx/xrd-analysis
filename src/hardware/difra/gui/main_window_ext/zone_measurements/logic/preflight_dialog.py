import logging

from PyQt5.QtWidgets import (
    QCheckBox,
    QDialog,
    QDialogButtonBox,
    QLabel,
    QMessageBox,
    QVBoxLayout,
)

from hardware.difra.gui.container_api import get_schema

logger = logging.getLogger(__name__)


class PreflightDialog(QDialog):
    """Mandatory pre-capture checklist - confirm technical container is valid.
    """

    def __init__(
        self,
        parent,
        session_manager=None,
    ):
        super().__init__(parent)
        self.setWindowTitle("Preflight Check")
        self.setModal(True)
        self.setMinimumWidth(540)
        self.session_manager = session_manager
        self.technical_h5_path = None

        self._setup_ui()
        self._revalidate()

    def _setup_ui(self):
        layout = QVBoxLayout(self)

        # Technical data info (from session)
        self.h5_label = QLabel("")
        self.h5_label.setWordWrap(True)
        self.h5_label.setStyleSheet("color: #222;")
        layout.addWidget(self.h5_label)
        
        # Confirmation checkbox
        self.h5_confirm = QCheckBox(
            "I confirm the technical data in this session is valid for these measurements."
        )
        layout.addWidget(self.h5_confirm)

        # Buttons
        self.buttons = QDialogButtonBox(QDialogButtonBox.Cancel | QDialogButtonBox.Ok)
        self.buttons.button(QDialogButtonBox.Ok).setText("Continue")
        self.buttons.button(QDialogButtonBox.Ok).setEnabled(False)
        self.buttons.rejected.connect(self.reject)
        self.buttons.accepted.connect(self._accept_if_valid)
        layout.addWidget(self.buttons)

        # Wire checkbox to revalidate
        self.h5_confirm.toggled.connect(self._revalidate)

    # --- Validation helpers ---
    def _detect_h5(self):
        """Get technical HDF5 container from session."""
        if not self.session_manager or not self.session_manager.is_session_active():
            return None
        
        try:
            # Get session path and look for technical data
            import h5py
            session_path = self.session_manager.session_path
            if not session_path:
                return None
            
            # Check if session has technical data group
            with h5py.File(session_path, 'r') as f:
                schema = get_schema(
                    getattr(self.session_manager, "config", None)
                    if self.session_manager
                    else None
                )
                candidates = []
                for attr in ("GROUP_CALIBRATION_SNAPSHOT", "GROUP_TECHNICAL", "GROUP_CALIBRATION"):
                    value = getattr(schema, attr, None)
                    if isinstance(value, str) and value:
                        candidates.append(value)
                candidates.extend(["/technical", "technical"])
                for group_name in candidates:
                    if group_name in f:
                        # Technical data exists in session - it was copied from technical container
                        self.technical_h5_path = session_path
                        return session_path
        except Exception as exc:
            logger.debug("Preflight technical-data detection failed: %s", exc)
        
        self.technical_h5_path = None
        return None

    def _revalidate(self):
        # Technical HDF5 container
        h5 = self._detect_h5()
        icon_h5 = "[OK]" if h5 else "[MISSING]"
        
        if h5:
            # Show session container name with technical data indicator
            container_name = h5.name if hasattr(h5, 'name') else str(h5).split('/')[-1]
            self.h5_label.setText(
                f"{icon_h5} Technical data available in session: {container_name}"
            )
        else:
            self.h5_label.setText(
                f"{icon_h5} No technical data found in active session"
            )

        # Enable Continue only if H5 confirmed and exists
        h5_confirmed = self.h5_confirm.isChecked()
        h5_exists = h5 is not None
        enable = h5_confirmed and h5_exists
        self.buttons.button(QDialogButtonBox.Ok).setEnabled(enable)


    def _accept_if_valid(self):
        if self.buttons.button(QDialogButtonBox.Ok).isEnabled():
            self.accept()
        else:
            QMessageBox.warning(self, "Preflight", "Please complete all confirmations.")
