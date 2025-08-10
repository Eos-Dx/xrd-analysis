"""Ulster XRD Analysis Package."""

# Global path setup to ensure xrdanalysis module is available
import sys
from pathlib import Path


def setup_xrdanalysis_path():
    """Add the src directory to Python path to access xrdanalysis module."""
    # Navigate from Ulster package to src directory
    # Current path: .../src/hardware/Ulster/__init__.py
    # Target path: .../src/
    src_path = Path(__file__).parent.parent.parent
    src_path_str = str(src_path)

    if src_path_str not in sys.path:
        sys.path.insert(0, src_path_str)

    return src_path_str


# Set up path immediately when Ulster package is imported
setup_xrdanalysis_path()
