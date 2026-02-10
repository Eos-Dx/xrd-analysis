"""Container loader with automatic version detection."""

from pathlib import Path
from typing import Union

import h5py


def detect_version(file_path: Union[str, Path]) -> str:
    """Detect container schema version from HDF5 file.
    
    Args:
        file_path: Path to HDF5 container
        
    Returns:
        Version string (e.g., "0.1", "1.0")
        
    Raises:
        ValueError: If version cannot be detected
    """
    try:
        with h5py.File(file_path, "r") as f:
            # Try to read schema_version attribute
            version = f.attrs.get("schema_version", None)
            
            if version is not None:
                # Convert to string if needed
                if isinstance(version, bytes):
                    version = version.decode('utf-8')
                return str(version)
            
            # Fallback: check for v0.1 indicators
            if "sample_id" in f.attrs and "session_id" in f.attrs:
                return "0.1"
            elif "container_id" in f.attrs and "/technical" in f:
                return "0.1"
                
            raise ValueError("Cannot detect schema version from container")
            
    except Exception as e:
        raise ValueError(f"Failed to detect version: {e}")


def open_container(file_path: Union[str, Path], version: str = None, validate: bool = True):
    """Open HDF5 container with automatic version detection.
    
    Args:
        file_path: Path to HDF5 container
        version: Optional explicit version (e.g., "0.1"). If None, auto-detect.
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
        container = open_container('session.h5', version='0.1', validate=False)
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
    
    # Normalize version string
    version = str(version).replace(".", "_")  # "0.1" -> "0_1"
    
    # Map versions
    # v0.1 (beta) corresponds to schema_version 1.0 from DIFRA
    if version in ["0_1", "1_0"]:
        version = "0_1"
    
    # Load appropriate version module
    if version == "0_1":
        from .v0_1 import SessionContainer, TechnicalContainer, utils
        
        # Detect container type
        info = utils.get_container_info(str(file_path))
        container_type = info.get("container_type", "").lower()
        
        if container_type == "session":
            return SessionContainer.open(file_path, validate=validate)
        elif container_type == "technical":
            return TechnicalContainer.open(file_path, validate=validate)
        else:
            raise ValueError(f"Unknown container type: {container_type}")
    
    else:
        raise ValueError(f"Unsupported container version: {version}")
