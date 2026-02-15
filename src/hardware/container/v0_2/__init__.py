"""HDF5/NeXus Container v0.2 - DIFRA Data Model.

Implements NeXus-oriented technical and session containers:
- NXroot/NXentry layout
- NXdifra_technical and NXdifra_session app definitions
- Calibration snapshot embedded in session container
"""

__version__ = "0.2.0"

from . import container_manager
from . import measurement_counter
from . import reader
from . import schema
from . import session_container
from . import technical_container
from . import technical_validator
from . import utils
from . import validator
from . import writer
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
