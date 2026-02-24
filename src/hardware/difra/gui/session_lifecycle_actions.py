"""Higher-level session lifecycle workflows shared by GUI mixins."""

from dataclasses import dataclass, field
from pathlib import Path
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


class SessionLifecycleActions:
    """Shared lifecycle actions used by session-related GUI flows."""

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

                explicit_session_id = session_id_by_path.get(str(candidate))
                archived_path = SessionLifecycleService.archive_session_container(
                    session_path=candidate,
                    session_id=explicit_session_id,
                    archive_folder=archive_folder,
                )
                result.archived_paths.append(archived_path)
                result.moved += 1

                if export_old_format:
                    try:
                        summary = SessionOldFormatExporter.export_from_session_container(
                            archived_path,
                            config=config,
                            archive_folder=archive_folder,
                        )
                        result.old_format_paths.append(summary.export_dir)
                    except Exception as exc:
                        result.old_format_failed.append(
                            f"{candidate.name}: {exc}"
                        )

                if was_active:
                    result.archived_active_session = True
            except Exception as exc:
                result.failed.append(f"{candidate.name}: {exc}")

        return result
