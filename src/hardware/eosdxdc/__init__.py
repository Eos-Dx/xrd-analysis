"""Compatibility alias for legacy imports expecting hardware.EosDxDc.*

Maps to hardware.eosdxdc subpackages.
"""

import importlib
import sys

# Import root package to ensure it's present
import hardware.eosdxdc as _e

# Map common subpackages
for _name in ("gui", "hardware", "utils"):
    sys.modules[__name__ + "." + _name] = importlib.import_module(
        "hardware.eosdxdc." + _name
    )
