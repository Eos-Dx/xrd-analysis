"""Top-level hardware package.

Provides access to difra and compatibility shims.
"""

# Provide legacy shim for hardware.xystages
# Ensure difra subpackage is discoverable
from . import difra  # noqa: F401
from . import xystages as xystages  # noqa: F401
