"""HDF5 Container Management for XRD Analysis.

This package provides versioned HDF5 container interfaces for storing
and retrieving X-ray diffraction data with complete metadata.

Versioning:
- v0.1 (beta): Initial DIFRA HDF5 Data Model implementation

Usage:
    # Auto-detect version and open container
    from hardware.container import open_container
    container = open_container('path/to/file.h5')
    
    # Manual version specification
    from hardware.container.v0_1 import SessionContainer
    container = SessionContainer.open('path/to/file.h5')
"""

__version__ = "0.1.0-beta"

from .loader import open_container
from .manager import is_container_locked, lock_container, unlock_container

__all__ = [
    "open_container",
    "is_container_locked",
    "lock_container",
    "unlock_container",
    "__version__",
]
