"""Shared helpers for session container lifecycle actions.

This module centralizes lock/archive behavior so UI mixins can delegate
domain actions instead of duplicating file-management logic.
"""

import shutil
import time
from pathlib import Path
from typing import Any, Dict, Optional


class SessionLifecycleService:
    """Utility methods for lock/archive workflow of session containers."""

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
        target_dir = resolved_archive / f"{sid}_{archive_stamp}"
        suffix = 1
        while target_dir.exists():
            suffix += 1
            target_dir = resolved_archive / f"{sid}_{archive_stamp}_{suffix}"
        target_dir.mkdir(parents=True, exist_ok=False)

        destination = target_dir / source.name
        shutil.move(str(source), str(destination))
        return destination
