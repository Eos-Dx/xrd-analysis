"""HDF5 Container v0.1 (beta) - DIFRA Data Model.

This version implements the complete DIFRA HDF5 Data Model specification
with support for:
- Session containers (sample-oriented, self-sufficient)
- Technical containers (calibration data)
- Images, zones, and mapping
- Point-centric measurements
- Analytical measurements
- Global measurement counter
"""

__version__ = "0.1.0-beta"

from . import schema
from . import utils
from . import writer
from . import session_container
from . import validator
from . import technical_validator
from . import technical_container
from . import measurement_counter
from . import container_manager
from .reader import SessionContainer, TechnicalContainer

__all__ = [
    "schema",
    "utils", 
    "writer",
    "validator",
    "technical_validator",
    "technical_container",
    "measurement_counter",
    "container_manager",
    "session_container",
    "SessionContainer",
    "TechnicalContainer",
    "__version__",
]
