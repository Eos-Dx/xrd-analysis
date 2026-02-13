"""General container lifecycle helpers that dispatch by version."""

from pathlib import Path
from typing import Optional, Union

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

