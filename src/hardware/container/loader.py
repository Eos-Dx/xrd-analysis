"""Container loader with automatic version detection."""

from pathlib import Path
import tempfile
from typing import Union
import zipfile

import h5py

from .registry import load_version_module, normalize_version


def detect_version(file_path: Union[str, Path]) -> str:
    """Detect container schema version from HDF5 file.
    
    Args:
        file_path: Path to HDF5 container
        
    Returns:
        Version string (e.g., "0.2")
        
    Raises:
        ValueError: If version cannot be detected
    """
    try:
        with h5py.File(file_path, "r") as f:
            # Read schema_version attribute
            version = f.attrs.get("schema_version", None)
            
            if version is not None:
                # Convert to string if needed
                if isinstance(version, bytes):
                    version = version.decode('utf-8')
                return str(version)
            
            # v0.2 fallback marker checks (NeXus + entry definition)
            if (
                f.attrs.get("NX_class") == "NXroot"
                and "/entry" in f
                and "/entry/definition" in f
            ):
                return "0.2"

            raise ValueError("Cannot detect schema version from container")
            
    except Exception as e:
        raise ValueError(f"Failed to detect version: {e}")


def open_container(file_path: Union[str, Path], version: str = None, validate: bool = True):
    """Open HDF5 container with automatic version detection.
    
    Args:
        file_path: Path to HDF5 container
        version: Optional explicit version (e.g., "0.2"). If None, auto-detect.
        validate: Whether to validate container structure
        
    Returns:
        SessionContainer or TechnicalContainer instance
        
    Raises:
        ValueError: If version is unsupported or detection fails
        FileNotFoundError: If file doesn't exist
        
    Examples:
        # Auto-detect version
        container = open_container('session.h5')
        
        # Explicit version with validation disabled
        container = open_container('session.nxs.h5', version='0.2', validate=False)
    """
    file_path = Path(file_path)
    
    if not file_path.exists():
        raise FileNotFoundError(f"Container not found: {file_path}")
    
    # Detect version if not provided
    if version is None:
        try:
            version = detect_version(file_path)
        except ValueError as e:
            raise ValueError(
                f"Cannot auto-detect container version. "
                f"Please specify version explicitly. Error: {e}"
            )
    
    normalized = normalize_version(version)
    version_module = load_version_module(normalized)
    utils = version_module.utils
    session_cls = version_module.SessionContainer
    technical_cls = version_module.TechnicalContainer

    # Detect container type
    info = utils.get_container_info(str(file_path))
    container_type = info.get("container_type", "").lower()

    if container_type == "session":
        return session_cls.open(file_path, validate=validate)
    if container_type == "technical":
        return technical_cls.open(file_path, validate=validate)
    raise ValueError(f"Unknown container type: {container_type}")


def open_container_bundle(
    bundle_file: Union[str, Path],
    extract_to: Union[str, Path] = None,
    version: str = None,
    validate: bool = True,
):
    """Open a container from ZIP bundle.

    Expects at least one ``.h5`` file in the ZIP. If multiple are present,
    the first path-sorted match is used.
    """
    bundle_path = Path(bundle_file)
    if not bundle_path.exists():
        raise FileNotFoundError(f"Bundle not found: {bundle_path}")

    if extract_to is None:
        extract_dir = Path(tempfile.mkdtemp(prefix="difra_bundle_"))
    else:
        extract_dir = Path(extract_to)
        extract_dir.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(bundle_path, "r") as zf:
        zf.extractall(extract_dir)

    h5_files = sorted(
        (candidate for candidate in extract_dir.rglob("*.h5") if candidate.is_file()),
        key=lambda candidate: (len(candidate.parts), candidate.as_posix()),
    )
    if not h5_files:
        raise ValueError(f"No .h5 container found in bundle: {bundle_path}")

    return open_container(h5_files[0], version=version, validate=validate)
