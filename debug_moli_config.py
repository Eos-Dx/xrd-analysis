"""
Debug script to check Moli machine configuration loading.
Run this to see what's being loaded and why the stage isn't initializing.
"""

import json
import sys
import os

# Add paths
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src', 'hardware'))

# Load Moli config
config_path = r"C:\dev\xrd-analysis\src\hardware\eosdxdc\resources\config\setups\Ulster (Moli).json"

print("=" * 70)
print("MOLI MACHINE CONFIGURATION DEBUG")
print("=" * 70)

with open(config_path, 'r') as f:
    config = json.load(f)

print(f"\n📋 Setup Name: {config.get('name')}")
print(f"\n🔧 DEV Mode: {config.get('DEV', 'Not set (will default to True)')}")

# Check detectors
print("\n" + "=" * 70)
print("DETECTORS")
print("=" * 70)

detectors = config.get('detectors', [])
print(f"\nTotal detectors configured: {len(detectors)}")

for det in detectors:
    print(f"\n  • {det['alias']} ({det['type']})")
    print(f"    ID: {det['id']}")
    if det['type'] == 'Pixet':
        print(f"    SDK Path: {det.get('pixet_sdk_path', 'Not specified')}")

print(f"\nActive detectors (production): {config.get('active_detectors', [])}")
print(f"Active detectors (DEV mode): {config.get('dev_active_detectors', [])}")

# Check stages
print("\n" + "=" * 70)
print("TRANSLATION STAGES")
print("=" * 70)

stages = config.get('translation_stages', [])
print(f"\nTotal stages configured: {len(stages)}")

for stage in stages:
    print(f"\n  • {stage['alias']} ({stage['type']})")
    print(f"    ID: {stage['id']}")
    if stage['type'] == 'Marlin':
        settings = stage.get('settings', {})
        print(f"    Limits X: {settings.get('limits_mm', {}).get('x')}")
        print(f"    Limits Y: {settings.get('limits_mm', {}).get('y')}")
        print(f"    Baudrate: {settings.get('baudrate', 'Not specified')}")

print(f"\nActive stages (production): {config.get('active_translation_stages', [])}")
print(f"Active stages (DEV mode): {config.get('dev_active_stages', [])}")

# Simulate what hardware_control.py does
print("\n" + "=" * 70)
print("WHAT WILL BE INITIALIZED")
print("=" * 70)

dev_mode = config.get('DEV', True)
print(f"\nDEV Mode: {dev_mode}")

# Detectors
selected_det_ids = (
    config.get('dev_active_detectors', [])
    if dev_mode
    else config.get('active_detectors', [])
)
selected_dets = [d for d in detectors if d['id'] in selected_det_ids]

print(f"\n🔍 Will try to initialize {len(selected_dets)} detector(s):")
for det in selected_dets:
    print(f"  ✓ {det['alias']} ({det['type']}) - ID: {det['id']}")

# Stages
selected_stage_ids = (
    config.get('dev_active_stages', [])
    if dev_mode
    else config.get('active_translation_stages', [])
)

print(f"\n🔍 Looking for stage with ID in: {selected_stage_ids}")

selected_stage = next(
    (s for s in stages if s['id'] in selected_stage_ids), None
)

if selected_stage:
    print(f"  ✓ Will initialize: {selected_stage['alias']} ({selected_stage['type']}) - ID: {selected_stage['id']}")
else:
    print(f"  ✗ NO STAGE FOUND! No stage has ID matching {selected_stage_ids}")
    print(f"\n  Available stage IDs in config:")
    for s in stages:
        print(f"    - {s['id']} ({s['alias']})")

print("\n" + "=" * 70)
print("RECOMMENDATIONS")
print("=" * 70)

if not selected_stage:
    if dev_mode:
        print("\n⚠️  ISSUE: DEV mode is enabled but dev_active_stages points to:")
        print(f"     {config.get('dev_active_stages')}")
        print("\n💡 SOLUTIONS:")
        print("   1. Set DEV: false in the config to use real hardware")
        print("   2. OR change dev_active_stages to [\"COM4\"] to test Marlin stage in DEV mode")
    else:
        print("\n⚠️  ISSUE: Production mode but active_translation_stages doesn't match any stage ID")
        print("\n💡 SOLUTION: Check that active_translation_stages matches a stage ID")

if selected_dets:
    for det in selected_dets:
        if det['type'] == 'Pixet':
            sdk_path = det.get('pixet_sdk_path', '')
            if sdk_path and not os.path.exists(sdk_path):
                print(f"\n⚠️  WARNING: Pixet SDK path doesn't exist:")
                print(f"     {sdk_path}")
                print(f"   Detector '{det['alias']}' may fail to initialize")

print("\n" + "=" * 70)
print("To run the Moli machine with real hardware:")
print("  1. Make sure DEV is set to false in the config")
print("  2. Ensure COM4 is available and connected")
print("  3. Verify Pixet SDK path is correct")
print("=" * 70)
print()
