"""
Path setup helper to ensure xrdanalysis module can be imported.
This module should be imported before any other modules that depend on xrdanalysis.
"""

import sys
from pathlib import Path

def setup_xrdanalysis_path():
    """Add the src directory to Python path to access xrdanalysis module."""
    # Navigate from this file to the src directory
    # Current path: .../src/hardware/Ulster/core/measurement/_path_setup.py
    # Target path: .../src/
    src_path = Path(__file__).parent.parent.parent.parent.parent
    src_path_str = str(src_path)
    
    if src_path_str not in sys.path:
        sys.path.insert(0, src_path_str)
    
    return src_path_str

# Set up path immediately when this module is imported
setup_xrdanalysis_path()
