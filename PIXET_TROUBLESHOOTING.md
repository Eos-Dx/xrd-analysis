# Pixet Detector Troubleshooting Guide

## Issue: "No Pixet devices connected"

### Symptom
When initializing hardware, you see:
```
ERROR | hardware.difra.hardware.detectors | No Pixet devices connected
RuntimeError: Failed to initialize detector W0308
```

But PIXet Pro application can see and use the detector.

### Root Cause

**PIXet Pro holds exclusive access to the detector.**

When PIXet Pro is running (or was running and didn't release the device properly), it maintains exclusive USB access to the MiniPIX detector. The pypixet library in your Python application cannot access the detector while PIXet Pro has control.

### Diagnostic Result

Running `debug_pixet_devices.py` shows:
```
Number of devices detected: 1
Device 1:
  Full Name: 'FileDevice 0'
  Type: File Device (not a real detector)
```

The `FileDevice 0` is a fallback dummy device, not your actual MiniPIX detector.

## Solution

### Step 1: Close PIXet Pro Completely

1. **Close PIXet Pro** application if it's running
2. **Check Task Manager** (Ctrl+Shift+Esc) for any lingering PIXet processes
3. **End any** `PIXetPro.exe` or related processes
4. **Wait 5-10 seconds** for the USB device to be released

### Step 2: Verify Detection

Run the diagnostic script:
```bash
cd C:\dev\xrd-analysis
python debug_pixet_devices.py
```

**Expected output after closing PIXet Pro:**
```
Number of devices detected: 1
Device 1:
  Full Name: 'MiniPIX G08-W0308'  (or similar)
  Type: Physical Detector
  ✓ MATCH: Contains 'W0308'
```

### Step 3: Start Your Application

Once the diagnostic shows a physical detector:
1. Start your XRD analysis software
2. Select "Ulster (Moli)" setup
3. Press INIT
4. Both detector and stage should initialize successfully

## Common Issues

### Issue 1: Still Shows FileDevice After Closing PIXet Pro

**Solution:**
- Unplug the USB cable from the detector
- Wait 5 seconds
- Plug it back in
- Run diagnostic again

### Issue 2: Wrong Device ID in Config

If diagnostic shows a different device name (e.g., `'MiniPIX G08-W0299'` instead of W0308):

**Solution:** Update the config file:

Edit: `src/hardware/difra/resources/config/setups/Ulster (Moli).json`

```json
{
  "alias": "PRIMARY",
  "type": "Pixet",
  "id": "W0299",  // Change this to match your actual detector
  ...
}
```

The `id` field should contain a substring that uniquely identifies your detector.

### Issue 3: Detector Works in PIXet Pro but Not in Python

This is normal behavior. **You cannot use both simultaneously.** Choose one:

**Option A: Use PIXet Pro for testing**
- Close your Python application
- Open PIXet Pro

**Option B: Use Python application**
- Close PIXet Pro completely  
- Run your Python application

## Workflow for Daily Use

### Recommended Workflow:

1. **Power on detector** (if external power required)
2. **Connect USB cable**
3. **Wait 10 seconds** for Windows to recognize device
4. **Do NOT open PIXet Pro** if you want to use Python application
5. **Run diagnostic** (optional, to verify)
6. **Start your application**

### If You Need to Switch:

**From Python to PIXet Pro:**
1. Stop/close your Python application
2. Wait 5 seconds
3. Open PIXet Pro

**From PIXet Pro to Python:**
1. Close PIXet Pro completely
2. Check Task Manager for lingering processes
3. Wait 5 seconds
4. Start your Python application

## USB Device Reset (If Needed)

If the detector is stuck and won't release:

**Windows:**
1. Open Device Manager
2. Find "Imaging devices" or "Universal Serial Bus devices"
3. Right-click on MiniPIX device
4. Select "Disable device"
5. Wait 3 seconds
6. Right-click again, select "Enable device"
7. Wait 5 seconds
8. Try diagnostic again

## Technical Details

### Why This Happens

USB devices can only be accessed by one application at a time for exclusive operations like:
- Configuring acquisition parameters
- Triggering measurements
- Reading detector data

PIXet Pro opens the device in exclusive mode, preventing other applications from accessing it.

### pypixet Library Behavior

When no physical devices are available, pypixet creates a `FileDevice 0` as a fallback. This allows:
- Code to run without crashing
- Testing with simulated data
- Development without hardware

But `FileDevice 0` **cannot** perform actual measurements.

## Quick Reference

### Diagnostic Command
```bash
python C:\dev\xrd-analysis\debug_pixet_devices.py
```

### Expected Good Output
```
✓ pypixet module imported successfully
✓ pypixet.start() successful
Number of devices detected: 1
Device 1:
  Full Name: 'MiniPIX G08-W0308'
  Type: Physical Detector
  ✓ MATCH: Contains 'W0308'
```

### Expected Bad Output (PIXet Pro running)
```
Number of devices detected: 1
Device 1:
  Full Name: 'FileDevice 0'
  Type: File Device (not a real detector)
```

## Summary

✅ **DO:**
- Close PIXet Pro before starting Python app
- Run diagnostic to verify detector is accessible
- Wait a few seconds after closing PIXet Pro

❌ **DON'T:**
- Try to use PIXet Pro and Python simultaneously
- Assume detector is accessible just because PIXet Pro sees it
- Skip the diagnostic step if having issues

## Need More Help?

Run the diagnostic and share the output:
```bash
python debug_pixet_devices.py > pixet_diagnostic.txt
```

This will help identify the specific issue with your setup.
