"""Shared helpers for session container lifecycle actions.

This module centralizes lock/archive behavior so UI mixins can delegate
domain actions instead of duplicating file-management logic.
"""

import shutil
import time
from pathlib import Path
from typing import Any, Dict, Optional

import h5py


class SessionLifecycleService:
    """Utility methods for lock/archive workflow of session containers."""

    @staticmethod
    def _decode_attr(value: Any) -> str:
        if value is None:
            return ""
        if isinstance(value, bytes):
            return value.decode("utf-8", errors="replace")
        return str(value)

    @staticmethod
    def _safe_token(value: str, fallback: str = "unknown") -> str:
        token = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in (value or ""))
        token = token.strip("_")
        return token or fallback

    @classmethod
    def _resolve_operator_id(
        cls,
        session_path: Path,
        explicit_operator_id: Optional[str] = None,
    ) -> str:
        if explicit_operator_id:
            return cls._decode_attr(explicit_operator_id) or "unknown"

        try:
            with h5py.File(session_path, "r") as h5f:
                root_operator = cls._decode_attr(h5f.attrs.get("operator_id"))
                if root_operator:
                    return root_operator

                user_group = h5f.get("/entry/user")
                if user_group is not None:
                    group_operator = cls._decode_attr(user_group.attrs.get("operator_id"))
                    if group_operator:
                        return group_operator

                lock_operator = cls._decode_attr(h5f.attrs.get("locked_by"))
                if lock_operator:
                    return lock_operator
        except Exception:
            pass

        return "unknown"

    @staticmethod
    def resolve_archive_folder(
        config: Optional[Dict[str, Any]] = None,
        measurements_folder: Optional[Path] = None,
        session_path: Optional[Path] = None,
    ) -> Path:
        """Resolve archive folder using config-first, deterministic fallbacks."""
        cfg = config or {}
        configured = cfg.get("measurements_archive_folder") or cfg.get(
            "session_archive_folder"
        )
        if configured:
            return Path(configured)

        if measurements_folder is not None:
            return Path(measurements_folder).parent / "archive" / "measurements"

        if session_path is not None:
            sp = Path(session_path)
            return sp.parent.parent / "archive" / "measurements"

        return Path.home() / "difra_measurements" / "archive"

    @staticmethod
    def lock_container_if_needed(
        container_path: Path,
        container_manager: Any,
        user_id: Optional[str] = None,
    ) -> bool:
        """Lock container only when it is still unlocked.

        Returns True when lock was applied during this call.
        """
        path = Path(container_path)
        if container_manager.is_container_locked(path):
            return False
        container_manager.lock_container(path, user_id=user_id)
        return True

    @classmethod
    def archive_session_container(
        cls,
        session_path: Path,
        session_id: Optional[str] = None,
        operator_id: Optional[str] = None,
        archive_folder: Optional[Path] = None,
        config: Optional[Dict[str, Any]] = None,
        measurements_folder: Optional[Path] = None,
        timestamp: Optional[str] = None,
    ) -> Path:
        """Move a session container into the archive tree and return destination."""
        source = Path(session_path)
        resolved_archive = (
            Path(archive_folder)
            if archive_folder is not None
            else cls.resolve_archive_folder(
                config=config,
                measurements_folder=measurements_folder,
                session_path=source,
            )
        )
        resolved_archive.mkdir(parents=True, exist_ok=True)

        archive_stamp = timestamp or time.strftime("%Y%m%d_%H%M%S")
        sid = str(session_id or source.stem)
        operator = cls._resolve_operator_id(source, explicit_operator_id=operator_id)
        operator_token = cls._safe_token(operator, fallback="unknown")
        target_dir = resolved_archive / f"{sid}_{operator_token}_{archive_stamp}"
        suffix = 1
        while target_dir.exists():
            suffix += 1
            target_dir = resolved_archive / f"{sid}_{operator_token}_{archive_stamp}_{suffix}"
        target_dir.mkdir(parents=True, exist_ok=False)

        destination = target_dir / source.name
        shutil.move(str(source), str(destination))
        return destination
