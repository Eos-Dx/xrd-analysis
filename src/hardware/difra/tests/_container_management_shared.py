"""Comprehensive tests for container management features.

Tests:
- PONI distance parsing and validation
- Container locking (HDF5 + OS permissions)
- Archive management with user confirmation
- Primary/supplementary measurement marking
- Find active containers
- Session container lock checking
"""

import os
import stat
import tempfile
import zipfile
from pathlib import Path

import h5py
import numpy as np
import pytest

# Add project src to path
import sys
SRC_ROOT = Path(__file__).resolve().parents[3]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from hardware.container.v0_2 import (
    schema,
    writer,
    technical_container,
    container_manager,
)
from hardware.container.loader import open_container_bundle
from hardware.container.manager import create_container_bundle
