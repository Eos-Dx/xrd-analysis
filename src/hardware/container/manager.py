"""General container lifecycle helpers that dispatch by version."""

from pathlib import Path
from typing import Optional, Sequence, Union

from .loader import detect_version
from .registry import load_version_module


def _get_container_manager_module(
    container_file: Union[str, Path],
    version: Optional[str] = None,
):
    """Resolve versioned container manager module."""
    resolved_version = version or detect_version(container_file)
    version_module = load_version_module(resolved_version)
    return version_module.container_manager


def is_container_locked(container_file: Union[str, Path], version: Optional[str] = None) -> bool:
    """Check lock status for a technical container."""
    manager = _get_container_manager_module(container_file, version=version)
    return manager.is_container_locked(Path(container_file))


def lock_container(container_file: Union[str, Path], user_id: Optional[str] = None, version: Optional[str] = None) -> None:
    """Lock technical container using versioned manager."""
    manager = _get_container_manager_module(container_file, version=version)
    manager.lock_container(Path(container_file), user_id=user_id)


def unlock_container(container_file: Union[str, Path], version: Optional[str] = None) -> None:
    """Unlock technical container using versioned manager."""
    manager = _get_container_manager_module(container_file, version=version)
    manager.unlock_container(Path(container_file))


def create_container_bundle(
    container_file: Union[str, Path],
    source_folder: Optional[Union[str, Path]] = None,
    output_zip: Optional[Union[str, Path]] = None,
    include_patterns: Optional[Sequence[str]] = None,
    source_arcname: Optional[str] = None,
    version: Optional[str] = None,
) -> Path:
    """Create container ZIP bundle via versioned manager implementation."""
    manager = _get_container_manager_module(container_file, version=version)
    if not hasattr(manager, "create_container_bundle"):
        raise ValueError(f"Container version does not implement ZIP bundling: {version}")

    source_folder_path = Path(source_folder) if source_folder else None
    output_zip_path = Path(output_zip) if output_zip else None

    return manager.create_container_bundle(
        container_path=Path(container_file),
        source_folder=source_folder_path,
        output_zip=output_zip_path,
        include_patterns=list(include_patterns) if include_patterns is not None else None,
        source_arcname=source_arcname,
    )
