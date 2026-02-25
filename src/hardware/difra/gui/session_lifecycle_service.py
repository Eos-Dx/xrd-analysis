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

    @classmethod
    def _resolve_archive_metadata(
        cls,
        session_path: Path,
        explicit_session_id: Optional[str] = None,
        explicit_operator_id: Optional[str] = None,
    ) -> Dict[str, str]:
        """Resolve session/operator/sample/project tokens for archive naming."""
        data = {
            "session_id": cls._decode_attr(explicit_session_id) or "",
            "operator_id": cls._decode_attr(explicit_operator_id) or "",
            "sample_id": "",
            "project_id": "",
            "study_name": "",
        }

        try:
            with h5py.File(session_path, "r") as h5f:
                if not data["session_id"]:
                    data["session_id"] = cls._decode_attr(h5f.attrs.get("session_id"))
                if not data["operator_id"]:
                    data["operator_id"] = cls._decode_attr(h5f.attrs.get("operator_id"))
                data["sample_id"] = cls._decode_attr(h5f.attrs.get("sample_id"))
                data["project_id"] = cls._decode_attr(h5f.attrs.get("project_id"))
                data["study_name"] = cls._decode_attr(h5f.attrs.get("study_name"))

                if not data["operator_id"]:
                    user_group = h5f.get("/entry/user")
                    if user_group is not None:
                        data["operator_id"] = cls._decode_attr(
                            user_group.attrs.get("operator_id")
                        )
                if not data["sample_id"]:
                    sample_group = h5f.get("/entry/sample")
                    if sample_group is not None:
                        data["sample_id"] = cls._decode_attr(
                            sample_group.attrs.get("sample_id")
                        )
                if not data["project_id"]:
                    if data["study_name"]:
                        data["project_id"] = data["study_name"]
                    else:
                        sample_group = h5f.get("/entry/sample")
                        if sample_group is not None:
                            data["project_id"] = cls._decode_attr(
                                sample_group.attrs.get("project_id")
                            )
                if not data["operator_id"]:
                    data["operator_id"] = cls._decode_attr(h5f.attrs.get("locked_by"))
        except Exception:
            pass

        if not data["session_id"]:
            data["session_id"] = str(session_path.stem)
        if not data["operator_id"]:
            data["operator_id"] = "unknown"
        if not data["sample_id"]:
            data["sample_id"] = "UNKNOWN"
        if not data["project_id"]:
            data["project_id"] = "UNSPECIFIED"

        return data

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
        metadata = cls._resolve_archive_metadata(
            source,
            explicit_session_id=session_id,
            explicit_operator_id=operator_id,
        )
        sid_token = cls._safe_token(metadata.get("session_id"), fallback="session")
        operator_token = cls._safe_token(
            metadata.get("operator_id"), fallback="unknown"
        )
        sample_token = cls._safe_token(metadata.get("sample_id"), fallback="UNKNOWN")
        project_token = cls._safe_token(
            metadata.get("project_id"), fallback="UNSPECIFIED"
        )
        target_dir = resolved_archive / (
            f"{sid_token}_{operator_token}_{sample_token}_{project_token}_{archive_stamp}"
        )
        suffix = 1
        while target_dir.exists():
            suffix += 1
            target_dir = resolved_archive / (
                f"{sid_token}_{operator_token}_{sample_token}_{project_token}_{archive_stamp}_{suffix}"
            )
        target_dir.mkdir(parents=True, exist_ok=False)

        destination = target_dir / source.name
        shutil.move(str(source), str(destination))
        return destination
