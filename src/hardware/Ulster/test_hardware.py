#!/usr/bin/env python3
import sys
from pathlib import Path

# Add current directory to Python path for local imports
sys.path.insert(0, str(Path(__file__).resolve().parent))

# Set up path for xrdanalysis module directly
src_path = Path(__file__).resolve().parent.parent.parent
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

print("Current working directory:", Path.cwd())
print("Python path:")
for i, path in enumerate(sys.path[:10]):  # Show first 10 entries
    print(f"  {i}: {path}")

try:
    print(
        "\nTrying: from ulster_hardware.controllers.detector import DummyDetectorController"
    )
    from ulster_hardware.controllers.detector import DummyDetectorController

    print("✓ Success!")
except Exception as e:
    print(f"✗ Failed: {e}")

try:
    print("\nTrying: import ulster_hardware.controllers")
    import ulster_hardware.controllers

    print("✓ Success!")
except Exception as e:
    print(f"✗ Failed: {e}")

try:
    print("\nTrying: import ulster_hardware")
    import ulster_hardware

    print(f"Ulster hardware module: {ulster_hardware}")
    print(
        f"Ulster hardware file: {getattr(ulster_hardware, '__file__', 'No __file__')}"
    )
except Exception as e:
    print(f"✗ Failed: {e}")
