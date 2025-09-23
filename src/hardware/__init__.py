"""Top-level hardware package.

Provides access to eosdxdc and compatibility shims.
"""

# Provide legacy shim for hardware.xystages
# Ensure eosdxdc subpackage is discoverable
from . import eosdxdc  # noqa: F401
from . import xystages as xystages  # noqa: F401
