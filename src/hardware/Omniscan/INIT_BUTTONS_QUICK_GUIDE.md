# Dashboard Init Buttons - Quick Guide

## What Changed

The Dashboard page now has two new initialization buttons in a prominent blue panel.

---

## Before vs After

### BEFORE
```
Dashboard
├── Safety Interlocks | Hardware Status
├── ⓘ Info banner (measurements in Measurement tab)
└── (No hardware control options)
```

### AFTER
```
Dashboard
├── Safety Interlocks | Hardware Status
├── 🔧 Hardware Initialization (NEW!)
│   ├── [Init Detector] [Init Motion]
│   └── Instructions about ENABLE button requirement
├── ⓘ Info banner (measurements in Measurement tab)
└── (Notifications appear top-right)
```

---

## How to Use

### Quick Steps

1. **Navigate to Dashboard**
   - Click "Dashboard" tab after login
   - System status displays on left/right panels

2. **Initialize Detector** (if needed)
   - Go to GPIO control panel
   - Press physical **ENABLE button**
   - Immediately return to computer (20s window!)
   - Click **"Init Detector"** button
   - Wait for notification

3. **Initialize Motion** (if needed)
   - Repeat steps for Motion system
   - Go to GPIO panel → Press ENABLE button
   - Return to computer
   - Click **"Init Motion"** button
   - Wait for notification

4. **Verify Success**
   - Green notification = initialization complete
   - Red notification = issue needs fixing
   - Hardware Status panel updates

---

## Button Behavior

| State | Appearance | Clickable | What Happens |
|-------|------------|-----------|--------------|
| Ready | Blue button | YES | Click to start init |
| Initializing | Gray button + spinner | NO | Processing... wait |
| Success | Green notification | YES | Can click again |
| Error | Red notification | YES | Fix issue & retry |

---

## Safety Features

### Enable Button
- **Purpose:** Physical confirmation by operator
- **Duration:** 20-second window
- **Required for:** All device initialization
- **Location:** GPIO control panel (hardware board)
- **Error if missed:** "Enable button not active"

### Automatic Safety Checks
The system automatically verifies:
- ✓ Key switch is ON
- ✓ Radiation is SAFE (beam blocked)
- ✓ Cooling is OK
- ✓ Power is OK

If any check fails → Red error notification tells you what to fix

---

## Common Scenarios

### Scenario 1: Successful Init
```
✓ Press ENABLE button on GPIO panel
✓ Click "Init Detector"
✓ Green notification: "Detector initialized successfully"
✓ Done!
```

### Scenario 2: Forgot to Press ENABLE Button
```
✗ Click "Init Detector" directly (without ENABLE button)
✗ Red notification: "Enable button not active"
✗ Go press ENABLE button on GPIO panel
✓ Click "Init Detector" again within 20 seconds
✓ Green notification appears
```

### Scenario 3: Safety Issue
```
✗ Press ENABLE button
✗ Click "Init Detector"
✗ Red notification: "Key switch must be ON"
✗ Go turn ON the key switch on GPIO panel
✓ Press ENABLE button again
✓ Click "Init Detector"
✓ Green notification appears
```

---

## Real-World Timeline Example

**13:00:00** - Operator logs in
- Dashboard displays
- Sees detector status: OFF
- Sees motion status: OFF

**13:00:15** - Operator initiates initialization
- Goes to GPIO control panel
- Presses ENABLE button (20s countdown starts: 13:00:15-13:00:35)

**13:00:18** - Operator back at computer
- Clicks "Init Detector" button
- Button turns gray with loading spinner
- Orchestrator validates ENABLE button is active ✓

**13:00:20** - Backend processing
- gRPC sends InitializeDetector to hardware server
- Detector powers up, runs init sequence

**13:00:21** - Detector ready
- Green notification: "Detector initialized successfully"
- Button returns to blue
- Hardware Status shows: Detector - IDLE

**13:00:22** - Operator ready for motion
- Goes to GPIO panel again
- Presses ENABLE button (new 20s window)

**13:00:25** - Operator back at computer
- Clicks "Init Motion" button
- Waits for notification...

**13:00:27** - Motion ready
- Green notification: "Motion initialized successfully"
- Both devices now initialized
- Ready for calibration or measurement

---

## Error Messages & Fixes

| Error Message | What It Means | How to Fix |
|---------------|---------------|-----------|
| "Enable button not active" | You didn't press ENABLE on GPIO panel | Press ENABLE button, then click Init button within 20s |
| "Radiation is NOT SAFE" | Beam is not blocked | Move beam block into place, then retry |
| "Cooling is NOT OK" | Cooling system not running | Check cooling pump power, verify no blockages, retry |
| "Power is NOT OK" | Power distribution issue | Check PDU power switch, verify connections, retry |
| "Key switch must be ON" | Physical key switch OFF | Turn key switch ON on GPIO panel, then retry |

---

## File Changes

**Only one file was modified:**

`omniscan-ui/src/pages/Dashboard.tsx`

**What was added:**
1. Import statements for `api` and `Loading` components
2. `handleInitializeDevice()` function to call backend API
3. New blue panel with Init buttons and instructions
4. State management for loading/notifications

**NO changes needed to:**
- API client (already has `initializeDevice()` method)
- Store (already has `addNotification()` method)
- Other components

---

## Testing Checklist

- [ ] Dashboard loads with Init buttons visible
- [ ] "Init Detector" button clickable
- [ ] "Init Motion" button clickable
- [ ] Click disables button and shows spinner
- [ ] Success shows green notification
- [ ] Error shows red notification
- [ ] Can click both buttons independently
- [ ] Instructions text is clear and visible
- [ ] Buttons style matches other UI elements
- [ ] Loading spinner animates smoothly

---

## Troubleshooting

**Init button not working:**
- Check if session is valid (logged in)
- Verify orchestrator is running (localhost:8081)
- Check browser console for errors

**No notification appearing:**
- Verify notification system in store
- Check if API response is being returned correctly
- Look in browser dev tools Network tab

**Button stays gray:**
- Page might need refresh
- Try clicking again
- Clear browser cache if issue persists

---

## Next Steps

After successful initialization:
1. **Calibration Tab** - Calibrate detectors if needed
2. **Measurement Tab** - Start your measurements
3. **Dashboard** - Monitor device status anytime

---

## Summary

✅ **What you can do now:**
- Initialize Detector and Motion from Dashboard
- See real-time status and error messages
- Safely initialize with automatic safety checks
- One-click access to hardware control

✅ **Safety is built-in:**
- Requires physical ENABLE button press
- Verifies all interlocks before initialization
- Clear error messages guide operator
- 20-second window prevents accidents
