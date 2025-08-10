#!/usr/bin/env python3
"""Test script to verify all imports work correctly after restructure."""

import sys
from pathlib import Path

# Add current directory to Python path for local imports
sys.path.insert(0, str(Path(__file__).parent))

# Set up path for xrdanalysis module directly
# From Ulster directory, go up 2 levels to reach src directory
current_dir = Path(__file__).resolve().parent
src_path = current_dir.parent.parent
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))
print(f"Added src path to sys.path: {src_path}")

def test_imports():
    """Test all critical imports."""
    
    try:
        print("Testing utils.logging imports...")
        from utils.logging import get_module_logger, setup_logging
        print("✓ utils.logging imports successful")
    except Exception as e:
        print(f"✗ utils.logging import failed: {e}")
        return False
    
    try:
        print("Testing config.settings imports...")
        from config.settings import config_manager
        print("✓ config.settings imports successful")
    except Exception as e:
        print(f"✗ config.settings import failed: {e}")
        return False
        
    try:
        print("Testing core.measurement imports...")
        from core.measurement import MeasurementWorker
        print("✓ core.measurement imports successful")
    except Exception as e:
        print(f"✗ core.measurement import failed: {e}")
        return False
        
    try:
        print("Testing ulster_hardware.controllers imports...")
        from ulster_hardware.controllers.detector import DummyDetectorController
        print("✓ ulster_hardware.controllers imports successful")
    except Exception as e:
        print(f"✗ ulster_hardware.controllers import failed: {e}")
        return False
        
    try:
        print("Testing ulster_hardware.manager imports...")
        from ulster_hardware.manager import HardwareController
        print("✓ ulster_hardware.manager imports successful")
    except Exception as e:
        print(f"✗ ulster_hardware.manager import failed: {e}")
        return False
        
    # Test GUI imports (might fail if PyQt5 not available)
    try:
        print("Testing gui.app imports...")
        from gui.app import UlsterApp
        print("✓ gui.app imports successful")
    except ImportError as e:
        if "PyQt5" in str(e):
            print("⚠ gui.app import skipped (PyQt5 not available)")
        else:
            print(f"✗ gui.app import failed: {e}")
            return False
    except Exception as e:
        print(f"✗ gui.app import failed: {e}")
        return False
        
    print("\n🎉 All critical imports successful!")
    return True

if __name__ == "__main__":
    success = test_imports()
    sys.exit(0 if success else 1)
