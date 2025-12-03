"""
Debug script to check what Pixet devices are detected and their names.
Run this to see the exact device names returned by pypixet.
"""

import sys
import os

# Add Pixet SDK path
pixet_sdk_path = r"C:\Program Files\PIXet Pro"
if os.path.exists(pixet_sdk_path):
    sys.path.insert(0, pixet_sdk_path)
    # Add to Windows PATH for DLL loading
    os.environ['PATH'] = pixet_sdk_path + os.pathsep + os.environ.get('PATH', '')
    print(f"✓ Added Pixet SDK path to sys.path and Windows PATH: {pixet_sdk_path}")
else:
    print(f"✗ Pixet SDK path not found: {pixet_sdk_path}")
    print("\nPlease update the pixet_sdk_path variable in this script to match your installation.")
    sys.exit(1)

print("=" * 70)
print("PIXET DETECTOR DIAGNOSTIC")
print("=" * 70)

try:
    import pypixet
    print("\n✓ pypixet module imported successfully")
except ImportError as e:
    print(f"\n✗ Failed to import pypixet: {e}")
    print("\nMake sure PIXet Pro is installed and the SDK path is correct.")
    sys.exit(1)

print("\n" + "=" * 70)
print("STARTING PIXET")
print("=" * 70)

try:
    pypixet.start()
    print("✓ pypixet.start() successful")
except Exception as e:
    print(f"✗ Failed to start pypixet: {e}")
    sys.exit(1)

pixet = pypixet.pixet

print("\n" + "=" * 70)
print("DETECTING DEVICES")
print("=" * 70)

try:
    devices = pixet.devices()
    print(f"\nNumber of devices detected: {len(devices)}")
    
    if not devices:
        print("\n✗ No devices found!")
        print("   Make sure the detector is:")
        print("   1. Powered on")
        print("   2. Connected via USB")
        print("   3. Not being used by another application (close PIXet Pro)")
    elif len(devices) == 1 and "FileDevice" in devices[0].fullName():
        print("\n⚠️  WARNING: Only FileDevice detected (no physical detector)")
        print("\n   This means pypixet cannot access the physical detector.")
        print("\n   TROUBLESHOOTING STEPS:")
        print("   1. ✗ CLOSE PIXet Pro application completely")
        print("      PIXet Pro holds exclusive access to the detector")
        print("   2. ✓ Make sure USB cable is securely connected")
        print("   3. ✓ Check Device Manager for the detector")
        print("   4. ✓ Try unplugging and replugging the USB cable")
        print("   5. ✓ Restart this script after closing PIXet Pro")
        print("\n   After closing PIXet Pro, wait 5 seconds and run this script again.")
    else:
        print("\nDevices found:")
        for i, dev in enumerate(devices):
            print(f"\n  Device {i + 1}:")
            full_name = dev.fullName()
            print(f"    Full Name: '{full_name}'")
            
            # Check if it's a real device or FileDevice
            if "FileDevice" in full_name:
                print(f"    Type: File Device (not a real detector)")
            else:
                print(f"    Type: Physical Detector")
            
            # Check what the config is looking for
            config_id = "W0308"
            if config_id in full_name:
                print(f"    ✓ MATCH: Contains '{config_id}' - This is the detector your config is looking for!")
            else:
                print(f"    ✗ NO MATCH: Does not contain '{config_id}'")
                print(f"       Your config is looking for: '{config_id}'")
                print(f"       But device name is: '{full_name}'")
                
                # Suggest what to put in config
                if "FileDevice" not in full_name:
                    print(f"\n       💡 SOLUTION: Update your config 'id' to match part of this name")
                    print(f"          For example, if the name contains 'W0299' or 'MiniPIX', use that")
    
    print("\n" + "=" * 70)
    print("CONFIGURATION COMPARISON")
    print("=" * 70)
    
    print(f"\nYour Ulster (Moli).json config has:")
    print(f"  \"id\": \"W0308\"")
    
    if devices and "FileDevice" not in devices[0].fullName():
        real_name = devices[0].fullName()
        print(f"\nActual device name from PIXet:")
        print(f"  \"{real_name}\"")
        
        print(f"\n💡 RECOMMENDATION:")
        print(f"   Update the 'id' field in your config to match the actual device name.")
        print(f"   The 'id' should be a substring that uniquely identifies your detector.")
        
        # Try to extract a reasonable ID
        if "MiniPIX" in real_name:
            parts = real_name.split()
            for part in parts:
                if part.startswith("W") or "MiniPIX" in part:
                    print(f"\n   Suggested config 'id': \"{part}\"")
                    break
    
except Exception as e:
    print(f"\n✗ Error detecting devices: {e}")
    import traceback
    traceback.print_exc()

finally:
    print("\n" + "=" * 70)
    print("CLEANUP")
    print("=" * 70)
    
    try:
        pixet.exitPixet()
        pypixet.exit()
        print("✓ Pixet cleaned up successfully")
    except Exception as e:
        print(f"⚠ Warning during cleanup: {e}")

print("\n" + "=" * 70)
print("DIAGNOSTIC COMPLETE")
print("=" * 70)
print()
