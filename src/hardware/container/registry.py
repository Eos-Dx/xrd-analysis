"""Version registry for HDF5 container implementations."""

from dataclasses import dataclass
from importlib import import_module
from types import ModuleType


@dataclass(frozen=True)
class VersionSpec:
    """Mapping from normalized version key to implementation module."""

    normalized_version: str
    module_path: str


VERSION_REGISTRY = {
    "0_2": VersionSpec(
        normalized_version="0_2",
        module_path="hardware.container.v0_2",
    ),
}


def normalize_version(version: str) -> str:
    """Normalize version values like '0.2' -> '0_2'."""
    return str(version).strip().replace(".", "_")


def get_version_spec(version: str) -> VersionSpec:
    """Get registered version spec or raise ValueError."""
    normalized = normalize_version(version)
    spec = VERSION_REGISTRY.get(normalized)
    if spec is None:
        supported = sorted(VERSION_REGISTRY.keys())
        raise ValueError(
            f"Unsupported container version: {version} "
            f"(normalized: {normalized}, supported: {supported})"
        )
    return spec


def load_version_module(version: str) -> ModuleType:
    """Load and return implementation module for version."""
    spec = get_version_spec(version)
    return import_module(spec.module_path)
