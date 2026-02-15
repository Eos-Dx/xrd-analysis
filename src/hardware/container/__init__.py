"""HDF5 Container Management for XRD Analysis.

This package provides versioned HDF5 container interfaces for storing
and retrieving X-ray diffraction data with complete metadata.

Versioning:
- v0.2: NeXus-based DIFRA technical/session containers

Usage:
    # Auto-detect version and open container
    from hardware.container import open_container
    container = open_container('path/to/file.h5')
    
    # Manual version specification
    from hardware.container.v0_2 import SessionContainer
    container = SessionContainer.open('path/to/file.h5')
"""

__version__ = "0.2.0"

from .loader import open_container, open_container_bundle
from .manager import (
    create_container_bundle,
    is_container_locked,
    lock_container,
    unlock_container,
)

__all__ = [
    "open_container",
    "open_container_bundle",
    "create_container_bundle",
    "is_container_locked",
    "lock_container",
    "unlock_container",
    "__version__",
]
