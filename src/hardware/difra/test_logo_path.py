#!/usr/bin/env python3
"""Test script to verify logo paths are correctly resolved."""
import sys
from pathlib import Path

print("=" * 60)
print("Logo Path Resolution Test")
print("=" * 60)

# Simulate the path resolution from main_window_basic.py
script_path = Path(__file__).resolve()
print(f"Script path: {script_path}")

app_root = script_path.parent
print(f"App root: {app_root}")

logo_dir = app_root / "resources/images"
print(f"Logo directory: {logo_dir}")
print(f"Logo directory exists: {logo_dir.exists()}")

if sys.platform == 'win32':
    # Windows: use .ico format
    logo_path = logo_dir / "rick_final.ico"
    if not logo_path.exists():
        logo_path = logo_dir / "rick_final.png"  # Fallback to PNG
elif sys.platform == 'darwin':
    # macOS: use .icns format for proper dock icon display
    logo_path = logo_dir / "rick_final.icns"
    if not logo_path.exists():
        logo_path = logo_dir / "rick_final.png"  # Fallback to PNG
else:
    # Linux: use .png format
    logo_path = logo_dir / "rick_final.png"

print(f"\nPlatform: {sys.platform}")
print(f"Selected logo path: {logo_path}")
print(f"Logo file exists: {logo_path.exists()}")

if logo_path.exists():
    size = logo_path.stat().st_size
    print(f"Logo file size: {size:,} bytes ({size/1024:.1f} KB)")
else:
    print("ERROR: Logo file not found!")
    print("\nAvailable files in logo directory:")
    if logo_dir.exists():
        for f in logo_dir.iterdir():
            print(f"  - {f.name} ({f.stat().st_size:,} bytes)")
    else:
        print("  Logo directory does not exist!")

print("=" * 60)
