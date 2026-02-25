"""Higher-level session lifecycle workflows shared by GUI mixins."""

from dataclasses import dataclass, field
from fnmatch import fnmatch
from pathlib import Path
import shutil
from typing import Any, Dict, Iterable, List, Optional

from hardware.difra.gui.session_lifecycle_service import SessionLifecycleService
from hardware.difra.gui.session_old_format_exporter import SessionOldFormatExporter


@dataclass
class SendArchiveResult:
    """Result summary for batch send+archive workflow."""

    moved: int = 0
    failed: List[str] = field(default_factory=list)
    archived_paths: List[Path] = field(default_factory=list)
    archived_active_session: bool = False
    old_format_paths: List[Path] = field(default_factory=list)
    old_format_failed: List[str] = field(default_factory=list)
    cleaned_artifacts: int = 0


class SessionLifecycleActions:
    """Shared lifecycle actions used by session-related GUI flows."""

    DEFAULT_MEASUREMENT_CLEANUP_PATTERNS = [
        "*.txt",
        "*.dsc",
        "*.npy",
        "*.t3pa",
        "*.poni",
        "*_state.json",
    ]

    @classmethod
    def _archive_measurement_artifacts(
        cls,
        measurements_folder: Path,
        destination_folder: Path,
    ) -> int:
        """Move raw measurement artifacts into the same archive folder as session H5."""
        source = Path(measurements_folder)
        destination = Path(destination_folder)
        if not source.exists() or not source.is_dir():
            return 0

        moved = 0
        patterns = cls.DEFAULT_MEASUREMENT_CLEANUP_PATTERNS
        destination.mkdir(parents=True, exist_ok=True)

        for file_path in sorted(source.rglob("*")):
            if not file_path.is_file():
                continue

            rel = file_path.relative_to(source)
            rel_posix = rel.as_posix()
            if rel_posix.startswith("archive/"):
                continue

            if not any(
                fnmatch(file_path.name, pattern) or fnmatch(rel_posix, pattern)
                for pattern in patterns
            ):
                continue

            target = destination / rel
            target.parent.mkdir(parents=True, exist_ok=True)

            if target.exists():
                stem = target.stem
                suffix = target.suffix
                idx = 2
                while True:
                    alt = target.with_name(f"{stem}_{idx}{suffix}")
                    if not alt.exists():
                        target = alt
                        break
                    idx += 1

            try:
                shutil.move(str(file_path), str(target))
                moved += 1
            except Exception:
                pass

        return moved

    @classmethod
    def _cleanup_measurement_artifacts(
        cls,
        measurements_folder: Path,
    ) -> int:
        """Remove transient measurement artifacts after successful archive."""
        folder = Path(measurements_folder)
        if not folder.exists() or not folder.is_dir():
            return 0

        removed = 0
        patterns = cls.DEFAULT_MEASUREMENT_CLEANUP_PATTERNS
        for file_path in sorted(folder.rglob("*")):
            if not file_path.is_file():
                continue
            rel = file_path.relative_to(folder).as_posix()
            if rel.startswith("archive/"):
                continue
            if any(fnmatch(file_path.name, p) or fnmatch(rel, p) for p in patterns):
                try:
                    file_path.unlink()
                    removed += 1
                except Exception:
                    pass

        # Legacy path was temporary only; remove it entirely when present.
        grpc_folder = folder / "grpc_exposures"
        if grpc_folder.exists() and grpc_folder.is_dir():
            try:
                shutil.rmtree(grpc_folder)
            except Exception:
                pass

        # Best-effort cleanup of now-empty nested directories.
        dirs = sorted(
            [d for d in folder.rglob("*") if d.is_dir()],
            key=lambda d: len(d.parts),
            reverse=True,
        )
        for dir_path in dirs:
            if dir_path == folder:
                continue
            try:
                dir_path.rmdir()
            except OSError:
                continue

        return removed

    @staticmethod
    def finalize_session_container(
        session_path: Path,
        container_manager: Any,
        lock_user: Optional[str] = None,
    ) -> bool:
        """Ensure session container is locked and ready for archive/upload."""
        return SessionLifecycleService.lock_container_if_needed(
            container_path=Path(session_path),
            container_manager=container_manager,
            user_id=lock_user,
        )

    @classmethod
    def send_and_archive_session_containers(
        cls,
        container_paths: Iterable[Path],
        *,
        container_manager: Any,
        archive_folder: Path,
        active_session_path: Optional[Path] = None,
        lock_user: Optional[str] = None,
        session_ids: Optional[Dict[str, str]] = None,
        config: Optional[Dict[str, Any]] = None,
        export_old_format: bool = True,
    ) -> SendArchiveResult:
        """Lock (if needed) and archive selected session containers."""
        result = SendArchiveResult()
        active_resolved = (
            Path(active_session_path).resolve()
            if active_session_path is not None
            else None
        )

        session_id_by_path = session_ids or {}
        cleanup_folders = set()

        for container_path in container_paths:
            candidate = Path(container_path)
            try:
                if not candidate.exists():
                    continue

                was_active = False
                if active_resolved is not None:
                    try:
                        was_active = candidate.resolve() == active_resolved
                    except Exception:
                        was_active = False

                cls.finalize_session_container(
                    session_path=candidate,
                    container_manager=container_manager,
                    lock_user=lock_user,
                )

                if export_old_format:
                    try:
                        # Legacy bundle is reconstructed from locked session H5 before archival move.
                        summary = SessionOldFormatExporter.export_from_session_container(
                            candidate,
                            config=config,
                            archive_folder=archive_folder,
                        )
                        result.old_format_paths.append(summary.export_dir)
                    except Exception as exc:
                        result.old_format_failed.append(
                            f"{candidate.name}: {exc}"
                        )

                explicit_session_id = session_id_by_path.get(str(candidate))
                archived_path = SessionLifecycleService.archive_session_container(
                    session_path=candidate,
                    session_id=explicit_session_id,
                    archive_folder=archive_folder,
                )
                result.archived_paths.append(archived_path)
                result.moved += 1
                cls._archive_measurement_artifacts(
                    measurements_folder=candidate.parent,
                    destination_folder=archived_path.parent,
                )
                try:
                    cleanup_folders.add(str(candidate.parent.resolve()))
                except Exception:
                    cleanup_folders.add(str(candidate.parent))

                if was_active:
                    result.archived_active_session = True
            except Exception as exc:
                result.failed.append(f"{candidate.name}: {exc}")

        for folder_str in sorted(cleanup_folders):
            try:
                result.cleaned_artifacts += cls._cleanup_measurement_artifacts(
                    Path(folder_str)
                )
            except Exception as exc:
                result.failed.append(f"cleanup {folder_str}: {exc}")

        return result
