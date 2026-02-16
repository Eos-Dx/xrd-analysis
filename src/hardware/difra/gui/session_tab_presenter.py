"""Presenter/service helpers for Session tab data and rendering."""

from dataclasses import dataclass
import time
from pathlib import Path
from typing import Any, Dict, List, Sequence

import h5py
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QCheckBox, QHBoxLayout, QTableWidget, QTableWidgetItem, QWidget


@dataclass(frozen=True)
class ActiveSessionViewState:
    """View state used by Session tab active-session section."""

    info_text: str
    close_enabled: bool
    upload_enabled: bool


class SessionTabPresenter:
    """Helpers for reading session metadata and rendering Session tab tables."""

    @staticmethod
    def decode_attr(value):
        if isinstance(value, bytes):
            return value.decode("utf-8", errors="replace")
        return value

    @staticmethod
    def scan_pending_session_containers(measurements_folder: Path) -> List[Path]:
        folder = Path(measurements_folder)
        if not folder.exists():
            return []
        return sorted(
            [path for path in folder.glob("session_*.nxs.h5") if path.is_file()]
        )

    @staticmethod
    def scan_archived_session_containers(archive_folder: Path) -> List[Path]:
        folder = Path(archive_folder)
        if not folder.exists():
            return []
        return sorted(
            [path for path in folder.rglob("session_*.nxs.h5") if path.is_file()]
        )

    @classmethod
    def read_session_container_metadata(
        cls,
        container_path: Path,
        *,
        schema: Any,
        container_manager: Any,
    ) -> Dict[str, str]:
        """Read core session metadata used by queue/archive tables."""
        path = Path(container_path)
        info: Dict[str, str] = {
            "file_name": path.name,
            "path": str(path),
            "sample_id": "UNKNOWN",
            "study_name": "UNSPECIFIED",
            "operator_id": "UNKNOWN",
            "created": "",
            "status": "UNKNOWN",
            "session_id": "",
            "archived": "",
        }

        try:
            with h5py.File(path, "r") as h5f:
                info["sample_id"] = str(
                    cls.decode_attr(h5f.attrs.get(schema.ATTR_SAMPLE_ID, "UNKNOWN"))
                )
                info["study_name"] = str(
                    cls.decode_attr(
                        h5f.attrs.get(schema.ATTR_STUDY_NAME, "UNSPECIFIED")
                    )
                )
                info["operator_id"] = str(
                    cls.decode_attr(h5f.attrs.get(schema.ATTR_OPERATOR_ID, "UNKNOWN"))
                )
                info["created"] = str(
                    cls.decode_attr(
                        h5f.attrs.get(schema.ATTR_CREATION_TIMESTAMP, "")
                    )
                )
                info["session_id"] = str(
                    cls.decode_attr(h5f.attrs.get(schema.ATTR_SESSION_ID, ""))
                )
                locked = container_manager.is_container_locked(path)
                info["status"] = "LOCKED" if locked else "UNLOCKED"
        except Exception as exc:
            info["status"] = f"ERROR ({exc})"

        try:
            parent_name = path.parent.name
            if "_" in parent_name:
                info["archived"] = parent_name.rsplit("_", 1)[-1]
        except Exception:
            pass
        if not info["archived"]:
            info["archived"] = time.strftime(
                "%Y%m%d_%H%M%S", time.localtime(path.stat().st_mtime)
            )

        return info

    @classmethod
    def build_pending_rows(
        cls,
        measurements_folder: Path,
        *,
        schema: Any,
        container_manager: Any,
    ) -> List[Dict[str, str]]:
        """Return metadata rows for pending session queue."""
        return [
            cls.read_session_container_metadata(
                container_path=container_path,
                schema=schema,
                container_manager=container_manager,
            )
            for container_path in cls.scan_pending_session_containers(measurements_folder)
        ]

    @classmethod
    def build_archived_rows(
        cls,
        archive_folder: Path,
        *,
        schema: Any,
        container_manager: Any,
    ) -> List[Dict[str, str]]:
        """Return metadata rows for archived session list."""
        return [
            cls.read_session_container_metadata(
                container_path=container_path,
                schema=schema,
                container_manager=container_manager,
            )
            for container_path in cls.scan_archived_session_containers(archive_folder)
        ]

    @staticmethod
    def _readonly_item(value: Any) -> QTableWidgetItem:
        item = QTableWidgetItem(str(value))
        item.setFlags(Qt.ItemIsSelectable | Qt.ItemIsEnabled)
        return item

    @classmethod
    def populate_pending_table(
        cls,
        table: QTableWidget,
        rows: Sequence[Dict[str, str]],
    ) -> None:
        """Populate pending queue table from presenter row data."""
        table.setRowCount(0)
        for row_index, row_data in enumerate(rows):
            table.insertRow(row_index)

            checkbox = QCheckBox()
            checkbox_widget = QWidget()
            checkbox_layout = QHBoxLayout(checkbox_widget)
            checkbox_layout.setContentsMargins(0, 0, 0, 0)
            checkbox_layout.setAlignment(Qt.AlignCenter)
            checkbox_layout.addWidget(checkbox)
            table.setCellWidget(row_index, 0, checkbox_widget)

            for col, key in enumerate(
                [
                    "file_name",
                    "sample_id",
                    "study_name",
                    "operator_id",
                    "created",
                    "status",
                ],
                start=1,
            ):
                table.setItem(row_index, col, cls._readonly_item(row_data.get(key, "")))

            table.setItem(row_index, 7, cls._readonly_item(row_data.get("path", "")))

    @classmethod
    def populate_archive_table(
        cls,
        table: QTableWidget,
        rows: Sequence[Dict[str, str]],
    ) -> None:
        """Populate archive table from presenter row data."""
        table.setRowCount(0)
        for row_index, row_data in enumerate(rows):
            table.insertRow(row_index)
            for col, key in enumerate(
                [
                    "file_name",
                    "sample_id",
                    "study_name",
                    "operator_id",
                    "created",
                    "archived",
                ]
            ):
                table.setItem(row_index, col, cls._readonly_item(row_data.get(key, "")))

            table.setItem(row_index, 6, cls._readonly_item(row_data.get("path", "")))

    @staticmethod
    def format_active_session_info(info: Dict[str, Any]) -> str:
        """Format active session info as HTML for QLabel."""
        return (
            f"<b>Sample ID:</b> {info['sample_id']}<br>"
            f"<b>Study:</b> {info.get('study_name', 'UNSPECIFIED')}<br>"
            f"<b>Session ID:</b> {info['session_id']}<br>"
            f"<b>Operator:</b> {info['operator_id']}<br>"
            f"<b>Container:</b> {Path(info['session_path']).name}<br>"
            f"<b>Status:</b> {'🔒 Locked' if info['is_locked'] else '🔓 Unlocked'}"
        )

    @classmethod
    def build_active_session_view_state(
        cls, info: Dict[str, Any]
    ) -> ActiveSessionViewState:
        """Build button/text state for active session info panel."""
        if not info.get("active"):
            return ActiveSessionViewState(
                info_text="No active session",
                close_enabled=False,
                upload_enabled=False,
            )

        is_locked = bool(info["is_locked"])
        return ActiveSessionViewState(
            info_text=cls.format_active_session_info(info),
            close_enabled=not is_locked,
            upload_enabled=is_locked,
        )
