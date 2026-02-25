"""Non-UI workflow service for active session finalization."""

from dataclasses import dataclass
from fnmatch import fnmatch
import json
import shutil
import time
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple

import h5py

from hardware.container import create_container_bundle
from hardware.difra.gui.session_lifecycle_actions import SessionLifecycleActions
from hardware.difra.gui.session_lifecycle_service import SessionLifecycleService


@dataclass
class FinalizeSessionResult:
    """Result payload for active-session finalization workflow."""

    session_path: Path
    archive_dest: Path
    archived_count: int
    bundle_path: Optional[Path]
    state_json_embedded: bool
    lock_applied_now: bool


class SessionFinalizeWorkflow:
    """Finalize active session containers without UI dependencies."""

    DEFAULT_ARCHIVE_PATTERNS = [
        "*.txt",
        "*.dsc",
        "*.npy",
        "*.t3pa",
        "*.poni",
        "*_state.json",
    ]

    @staticmethod
    def _as_text(value: Any, default: str = "") -> str:
        if value is None:
            return default
        if isinstance(value, bytes):
            return value.decode("utf-8", errors="replace")
        return str(value)

    @staticmethod
    def _safe_token(value: str, fallback: str) -> str:
        token = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in (value or ""))
        token = token.strip("_")
        return token or fallback

    @classmethod
    def ensure_human_readable_metadata(
        cls,
        session_path: Path,
        logger: Optional[Any] = None,
    ) -> Dict[str, str]:
        """Ensure session keeps user-readable IDs and summary for archive usage."""
        with h5py.File(session_path, "a") as h5f:
            sample = cls._as_text(h5f.attrs.get("sample_id"), "unknown")
            study = cls._as_text(h5f.attrs.get("study_name"), "UNSPECIFIED")
            project = cls._as_text(h5f.attrs.get("project_id"), study)
            operator = cls._as_text(h5f.attrs.get("operator_id"), "unknown")
            machine = cls._as_text(h5f.attrs.get("machine_name"), "unknown")
            site = cls._as_text(h5f.attrs.get("site_id"), "unknown")
            session_id = cls._as_text(h5f.attrs.get("session_id"), "unknown")
            acquisition_date = cls._as_text(h5f.attrs.get("acquisition_date"), "")
            created_at = cls._as_text(h5f.attrs.get("creation_timestamp"), "")

            summary = "\n".join(
                [
                    f"Sample ID: {sample}",
                    f"Project ID: {project}",
                    f"Study Name: {study}",
                    f"Operator ID: {operator}",
                    f"Machine: {machine}",
                    f"Site ID: {site}",
                    f"Session ID: {session_id}",
                    f"Acquisition Date: {acquisition_date}",
                    f"Created At: {created_at}",
                ]
            )

            h5f.attrs["project_id"] = project
            h5f.attrs["human_summary"] = summary
            if "/entry/sample" in h5f:
                h5f["/entry/sample"].attrs["project_id"] = project
            if "/entry/human_summary" in h5f:
                del h5f["/entry/human_summary"]
            if "/entry" in h5f:
                h5f.create_dataset(
                    "/entry/human_summary",
                    data=summary,
                    dtype=h5py.string_dtype(encoding="utf-8"),
                )

        if logger:
            logger.info(
                "Updated human-readable metadata in session container",
                session_path=str(session_path),
                sample_id=sample,
                project_id=project,
            )

        return {
            "sample_id": sample,
            "study_name": study,
            "project_id": project,
            "operator_id": operator,
            "machine_name": machine,
            "session_id": session_id,
        }

    @staticmethod
    def store_json_state_in_container(
        session_path: Path,
        measurements_folder: Path,
        sample_id: str,
        logger: Optional[Any] = None,
    ) -> bool:
        """Store state JSON content in session container as ``meta_json`` attr."""
        state_file = Path(measurements_folder) / f"{sample_id}_state.json"
        if not state_file.exists():
            if logger:
                logger.warning(f"State JSON file not found: {state_file}")
            return False

        try:
            with open(state_file, "r") as file_handle:
                state_data = json.load(file_handle)
            with h5py.File(session_path, "a") as h5f:
                h5f.attrs["meta_json"] = json.dumps(state_data)
            if logger:
                logger.info(
                    "Stored state JSON in container",
                    session_path=str(session_path),
                    state_file=str(state_file),
                )
            return True
        except Exception as exc:
            if logger:
                logger.error(
                    f"Failed to store state JSON in container: {exc}", exc_info=True
                )
            return False

    @classmethod
    def archive_measurement_files(
        cls,
        measurements_folder: Path,
        sample_id: str,
        session_id: Optional[str] = None,
        study_name: Optional[str] = None,
        project_id: Optional[str] = None,
        operator_id: Optional[str] = None,
        *,
        config: Optional[Dict[str, Any]] = None,
        include_patterns: Optional[Sequence[str]] = None,
        logger: Optional[Any] = None,
    ) -> Tuple[Path, int]:
        """Archive raw measurement files and return ``(dest_folder, file_count)``."""
        measurements_folder = Path(measurements_folder)
        archive_base = SessionLifecycleService.resolve_archive_folder(
            config=config,
            measurements_folder=measurements_folder,
        )

        timestamp = time.strftime("%Y%m%d_%H%M%S")
        date_token, time_token = timestamp.split("_", 1)
        session_token = cls._safe_token(session_id or "session", "session")
        operator_token = cls._safe_token(operator_id or "unknown", "unknown")
        sample_token = cls._safe_token(sample_id, "sample")
        project_token = cls._safe_token(
            project_id or study_name or "UNSPECIFIED",
            "UNSPECIFIED",
        )
        archive_name = (
            f"{session_token}_{operator_token}_{sample_token}_{project_token}_{date_token}_{time_token}"
        )
        archive_dest = archive_base / archive_name
        archive_dest.mkdir(parents=True, exist_ok=True)

        patterns = list(include_patterns) if include_patterns else cls.DEFAULT_ARCHIVE_PATTERNS
        archived_count = 0

        for file_path in sorted(measurements_folder.rglob("*")):
            if not file_path.is_file():
                continue

            relative_path = file_path.relative_to(measurements_folder)
            relative_str = relative_path.as_posix()
            matches_pattern = any(
                fnmatch(file_path.name, pattern) or fnmatch(relative_str, pattern)
                for pattern in patterns
            )
            if not matches_pattern:
                continue

            try:
                dest_path = archive_dest / relative_path
                dest_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(file_path), str(dest_path))
                archived_count += 1
            except Exception as exc:
                if logger:
                    logger.warning(f"Failed to archive {relative_path}: {exc}")

        if logger:
            logger.info(
                f"Archived {archived_count} measurement files",
                archive_folder=str(archive_dest),
            )
        return archive_dest, archived_count

    @staticmethod
    def archive_session_container_into_folder(
        session_path: Path,
        archive_folder: Path,
        logger: Optional[Any] = None,
    ) -> Path:
        """Move locked session container into the same archive folder as raw files."""
        source = Path(session_path)
        archive_folder = Path(archive_folder)
        archive_folder.mkdir(parents=True, exist_ok=True)

        destination = archive_folder / source.name
        suffix = 1
        while destination.exists():
            suffix += 1
            destination = archive_folder / f"{source.stem}_{suffix}{source.suffix}"

        shutil.move(str(source), str(destination))
        if logger:
            logger.info(
                "Archived locked session container",
                source=str(source),
                destination=str(destination),
            )
        return destination

    @staticmethod
    def create_session_bundle_zip(
        session_path: Path,
        archive_folder: Path,
        logger: Optional[Any] = None,
    ) -> Optional[Path]:
        """Create ZIP bundle for locked session + archived measurement files."""
        try:
            output_zip = Path(archive_folder).with_suffix(".zip")
            bundle_path = create_container_bundle(
                container_file=Path(session_path),
                source_folder=Path(archive_folder),
                output_zip=output_zip,
                source_arcname=Path(archive_folder).name,
            )
            if logger:
                logger.info("Created session ZIP bundle", bundle_path=str(bundle_path))
            return Path(bundle_path)
        except Exception as exc:
            if logger:
                logger.warning(f"Failed to create session ZIP bundle: {exc}")
            return None

    @classmethod
    def finalize_session(
        cls,
        *,
        session_path: Path,
        measurements_folder: Path,
        sample_id: str,
        container_manager: Any,
        lock_user: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        logger: Optional[Any] = None,
        include_patterns: Optional[Sequence[str]] = None,
    ) -> FinalizeSessionResult:
        """Run the full active-session finalization workflow."""
        readable_meta = cls.ensure_human_readable_metadata(
            session_path=session_path,
            logger=logger,
        )

        state_json_embedded = cls.store_json_state_in_container(
            session_path=session_path,
            measurements_folder=measurements_folder,
            sample_id=sample_id,
            logger=logger,
        )

        lock_applied_now = SessionLifecycleActions.finalize_session_container(
            session_path=session_path,
            container_manager=container_manager,
            lock_user=lock_user,
        )

        archive_dest, archived_count = cls.archive_measurement_files(
            measurements_folder=measurements_folder,
            sample_id=readable_meta.get("sample_id") or sample_id,
            session_id=readable_meta.get("session_id"),
            study_name=readable_meta.get("study_name"),
            project_id=readable_meta.get("project_id"),
            operator_id=readable_meta.get("operator_id"),
            config=config,
            include_patterns=include_patterns,
            logger=logger,
        )

        bundle_path = cls.create_session_bundle_zip(
            session_path=session_path,
            archive_folder=archive_dest,
            logger=logger,
        )

        archived_session_path = cls.archive_session_container_into_folder(
            session_path=session_path,
            archive_folder=archive_dest,
            logger=logger,
        )

        return FinalizeSessionResult(
            session_path=Path(archived_session_path),
            archive_dest=archive_dest,
            archived_count=archived_count,
            bundle_path=bundle_path,
            state_json_embedded=state_json_embedded,
            lock_applied_now=lock_applied_now,
        )
