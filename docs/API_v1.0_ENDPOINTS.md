# DiFRA Hardware Server API v1.0 - Endpoint Documentation

**Version:** 1.0.0  
**Framework:** FastAPI  
**Base URL:** `http://<server-ip>:8000/api/v1`  
**Authentication:** None (v1.0)  
**Real-time Protocol:** WebSocket

---

## Table of Contents

1. [Overview](#overview)
2. [Architecture Decisions](#architecture-decisions)
3. [Hardware Management](#hardware-management)
4. [Configuration Management](#configuration-management)
5. [Stage Control](#stage-control)
6. [Detector Control](#detector-control)
7. [Zone Measurements](#zone-measurements)
8. [Technical Measurements](#technical-measurements)
9. [WebSocket Real-time Channels](#websocket-real-time-channels)
10. [File Management](#file-management)
11. [Error Handling](#error-handling)
12. [Data Models](#data-models)

---

## Overview

The DiFRA Hardware Server API provides web-based control for X-ray diffraction hardware including:
- **XY Translation Stages** (Kinesis/Thorlabs, Marlin, Dummy)
- **CCD Detectors** (Pixet/MiniPIX, Dummy)
- **Automated Zone Measurements** (multi-point scanning)
- **Technical Measurements** (single-point with live preview)

### Key Features
- Manual hardware initialization/deinitialization
- Real-time updates via WebSocket (stage position, detector frames, progress)
- Config-driven setup with hot-swapping capability
- File-based data storage with Matador cloud integration (future)
- Conservative error handling with manual recovery options

---

## Architecture Decisions

Based on requirements gathered:

| Aspect | Decision | Rationale |
|--------|----------|-----------|
| **Framework** | FastAPI | Async support, WebSocket, auto docs, type hints |
| **Authentication** | None (v1.0) | Focus on functionality first |
| **Hardware Init** | Manual per session | Full control, matches current PyQt5 behavior |
| **Real-time** | WebSocket for all updates | Stage, detectors, progress, status, events |
| **Storage** | File-based | Simple, data flows to Matador cloud |
| **Image Handling** | Client-side UI, server storage | Client draws zones, server stores with measurements |
| **Measurement Modes** | Separate endpoints, mutex | Zone and Technical can't run concurrently |
| **Detector Config** | Config-based with API rewrite | Use active detectors from config, allow updates |
| **Parameters** | Per-measurement | Exposure, frames set per measurement request |
| **Validation** | Full (limits + exclusions + spacing) | Maximum safety |
| **Error Handling** | Pause for recovery, else abort | Conservative, manual intervention |

---

## Hardware Management

### Initialize Hardware
**Endpoint:** `POST /hardware/initialize`

Initialize all hardware (stages and detectors) based on current configuration.

**Request Body:**
```json
{
  "setup_name": "Ulster (Xena)",  // Optional, defaults to config default_setup
  "auto_home_stage": true          // Optional, default: true
}
```

**Response:**
```json
{
  "status": "success",
  "hardware_initialized": true,
  "stage": {
    "initialized": true,
    "alias": "XY_STAGE",
    "type": "Kinesis",
    "position": {"x": 0.0, "y": 0.0},
    "limits": {
      "x": [-14.0, 14.0],
      "y": [-14.0, 14.0]
    },
    "home_position": [6.13, -9.3],
    "load_position": [-13.9, -6.0]
  },
  "detectors": [
    {
      "initialized": true,
      "alias": "PRIMARY",
      "type": "Pixet",
      "id": "MiniPIX G08-W0299",
      "size": {"width": 256, "height": 256}
    },
    {
      "initialized": true,
      "alias": "SECONDARY",
      "type": "Pixet",
      "id": "MiniPIX G05-W0339",
      "size": {"width": 256, "height": 256}
    }
  ],
  "timestamp": "2026-01-27T19:20:00Z"
}
```

**Errors:**
- `500` - Hardware initialization failed (with detailed error per component)

---

### Deinitialize Hardware
**Endpoint:** `POST /hardware/deinitialize`

Safely deinitialize all hardware and release resources.

**Request Body:** None (or empty JSON `{}`)

**Response:**
```json
{
  "status": "success",
  "hardware_initialized": false,
  "message": "All hardware deinitialized successfully",
  "timestamp": "2026-01-27T19:21:00Z"
}
```

---

### Reinitialize Hardware
**Endpoint:** `POST /hardware/reinitialize`

Deinitialize and reinitialize hardware (useful for recovery from errors).

**Request Body:**
```json
{
  "setup_name": "Ulster (Xena)",  // Optional
  "auto_home_stage": true          // Optional
}
```

**Response:** Same as Initialize Hardware

---

### Get Hardware Status
**Endpoint:** `GET /hardware/status`

Get current hardware initialization and connection status.

**Response:**
```json
{
  "hardware_initialized": true,
  "stage": {
    "connected": true,
    "alias": "XY_STAGE",
    "type": "Kinesis",
    "position": {"x": 5.23, "y": -2.14},
    "moving": false
  },
  "detectors": [
    {
      "alias": "PRIMARY",
      "connected": true,
      "streaming": false
    },
    {
      "alias": "SECONDARY",
      "connected": true,
      "streaming": false
    }
  ],
  "measurement_active": false,
  "measurement_type": null,
  "timestamp": "2026-01-27T19:22:00Z"
}
```

---

## Configuration Management

### Get Current Configuration
**Endpoint:** `GET /config`

Retrieve the currently active configuration.

**Response:**
```json
{
  "setup_name": "Ulster (Xena)",
  "DEV": true,
  "detectors": [...],
  "active_detectors": ["MiniPIX G08-W0299", "MiniPIX G05-W0339"],
  "dev_active_detectors": ["DUMMY-0001", "DUMMY-0002"],
  "translation_stages": [...],
  "active_translation_stages": ["101370874"],
  "dev_active_stages": ["DUMMY-XY-000"],
  "default_folder": "E:\\test",
  "timestamp": "2026-01-27T19:23:00Z"
}
```

---

### Update Configuration
**Endpoint:** `PUT /config`

Update the configuration (writes to config file).

**Request Body:**
```json
{
  "DEV": false,
  "active_detectors": ["MiniPIX G08-W0299"],
  "default_folder": "D:\\measurements\\2026-01"
  // Any config fields to update
}
```

**Response:**
```json
{
  "status": "success",
  "message": "Configuration updated successfully",
  "config_file": "/path/to/config/setups/Ulster (Xena).json",
  "requires_reinit": true,  // If hardware-related changes made
  "timestamp": "2026-01-27T19:24:00Z"
}
```

---

### List Available Setups
**Endpoint:** `GET /config/setups`

List all available setup configuration files.

**Response:**
```json
{
  "setups": [
    {
      "name": "Ulster (Xena)",
      "file": "Ulster (Xena).json",
      "active": true
    },
    {
      "name": "Ulster (Moli)",
      "file": "Ulster (Moli).json",
      "active": false
    },
    {
      "name": "queen-mary",
      "file": "queen-mary.json",
      "active": false
    }
  ]
}
```

---

### Switch Setup
**Endpoint:** `POST /config/setup/switch`

Switch to a different setup configuration without server restart.

**Request Body:**
```json
{
  "setup_name": "Ulster (Moli)",
  "reinitialize_hardware": true  // Optional, default: false
}
```

**Response:**
```json
{
  "status": "success",
  "message": "Switched to setup: Ulster (Moli)",
  "hardware_reinitialized": true,
  "timestamp": "2026-01-27T19:25:00Z"
}
```

**Errors:**
- `400` - Cannot switch setup while measurement is active
- `404` - Setup file not found

---

### Get PONI Files
**Endpoint:** `GET /config/poni`

List available PONI calibration files.

**Response:**
```json
{
  "poni_files": [
    {
      "detector_alias": "PRIMARY",
      "file_name": "primary_detector_20250115.poni",
      "path": "/path/to/poni/primary_detector_20250115.poni",
      "default": true
    },
    {
      "detector_alias": "SECONDARY",
      "file_name": "secondary_detector_20250115.poni",
      "path": "/path/to/poni/secondary_detector_20250115.poni",
      "default": true
    }
  ]
}
```

---

### Upload PONI File
**Endpoint:** `POST /config/poni/upload`

Upload a new PONI calibration file.

**Request:** `multipart/form-data`
- `file`: PONI file
- `detector_alias`: Target detector (e.g., "PRIMARY")
- `set_as_default`: Boolean (optional)

**Response:**
```json
{
  "status": "success",
  "message": "PONI file uploaded successfully",
  "detector_alias": "PRIMARY",
  "file_name": "custom_calibration.poni",
  "path": "/path/to/poni/custom_calibration.poni",
  "set_as_default": true,
  "timestamp": "2026-01-27T19:26:00Z"
}
```

---

### Set PONI File for Detector
**Endpoint:** `POST /config/poni/set`

Set which PONI file to use for a detector.

**Request Body:**
```json
{
  "detector_alias": "PRIMARY",
  "poni_file_path": "/path/to/poni/custom_calibration.poni"
}
```

**Response:**
```json
{
  "status": "success",
  "message": "PONI file set for PRIMARY detector",
  "timestamp": "2026-01-27T19:27:00Z"
}
```

---

## Stage Control

### Move Stage to Position
**Endpoint:** `POST /stage/move`

Move the XY stage to absolute coordinates (mm).

**Request Body:**
```json
{
  "x": 5.5,
  "y": -3.2,
  "timeout": 20  // Optional, seconds, default: 20
}
```

**Response:**
```json
{
  "status": "success",
  "requested": {"x": 5.5, "y": -3.2},
  "actual": {"x": 5.501, "y": -3.199},
  "move_time": 1.234,  // seconds
  "timestamp": "2026-01-27T19:28:00Z"
}
```

**Errors:**
- `400` - Coordinates outside stage limits
- `409` - Stage busy (measurement in progress)
- `500` - Stage movement timeout or hardware error

---

### Get Stage Position
**Endpoint:** `GET /stage/position`

Get current XY stage position.

**Response:**
```json
{
  "position": {"x": 5.501, "y": -3.199},
  "moving": false,
  "timestamp": "2026-01-27T19:29:00Z"
}
```

---

### Home Stage
**Endpoint:** `POST /stage/home`

Home the XY stage (return to home position).

**Request Body:**
```json
{
  "timeout": 45  // Optional, seconds, default: 45
}
```

**Response:**
```json
{
  "status": "success",
  "position": {"x": 6.13, "y": -9.3},
  "home_time": 12.5,  // seconds
  "timestamp": "2026-01-27T19:30:00Z"
}
```

---

### Move to Load Position
**Endpoint:** `POST /stage/load`

Move stage to configured load position (sample loading area).

**Request Body:**
```json
{
  "timeout": 20  // Optional
}
```

**Response:**
```json
{
  "status": "success",
  "position": {"x": -13.9, "y": -6.0},
  "move_time": 3.2,
  "timestamp": "2026-01-27T19:31:00Z"
}
```

---

### Emergency Stop Stage
**Endpoint:** `POST /stage/emergency-stop`

Immediately halt all stage movement (emergency use).

**Request Body:** None

**Response:**
```json
{
  "status": "success",
  "message": "Emergency stop executed",
  "position": {"x": 5.123, "y": -2.456},  // Position where stopped
  "timestamp": "2026-01-27T19:32:00Z"
}
```

---

### Validate Stage Coordinates
**Endpoint:** `POST /stage/validate`

Pre-validate coordinates before moving (useful for validating measurement grids).

**Request Body:**
```json
{
  "points": [
    {"x": 5.5, "y": -3.2},
    {"x": 6.0, "y": -3.0},
    {"x": -15.0, "y": 2.0}  // This would be invalid
  ]
}
```

**Response:**
```json
{
  "all_valid": false,
  "results": [
    {
      "point": {"x": 5.5, "y": -3.2},
      "valid": true
    },
    {
      "point": {"x": 6.0, "y": -3.0},
      "valid": true
    },
    {
      "point": {"x": -15.0, "y": 2.0},
      "valid": false,
      "error": "X position -15.000 exceeds limits [-14.0, 14.0]"
    }
  ],
  "limits": {
    "x": [-14.0, 14.0],
    "y": [-14.0, 14.0]
  }
}
```

---

## Detector Control

### Get Detector List
**Endpoint:** `GET /detectors`

List all available detectors and their status.

**Response:**
```json
{
  "detectors": [
    {
      "alias": "PRIMARY",
      "type": "Pixet",
      "id": "MiniPIX G08-W0299",
      "active": true,
      "connected": true,
      "size": {"width": 256, "height": 256},
      "streaming": false
    },
    {
      "alias": "SECONDARY",
      "type": "Pixet",
      "id": "MiniPIX G05-W0339",
      "active": true,
      "connected": true,
      "size": {"width": 256, "height": 256},
      "streaming": false
    }
  ]
}
```

---

### Start Detector Stream
**Endpoint:** `POST /detectors/{alias}/stream/start`

Start live detector frame streaming (for Technical Measurements preview).

**Request Body:**
```json
{
  "exposure": 0.1,    // seconds
  "interval": 0.0,    // seconds between frames
  "frames": 1         // frames to integrate per capture
}
```

**Response:**
```json
{
  "status": "success",
  "detector_alias": "PRIMARY",
  "streaming": true,
  "websocket_channel": "detector_stream",
  "message": "Connect to WebSocket to receive frames",
  "timestamp": "2026-01-27T19:33:00Z"
}
```

**Note:** Frames are pushed via WebSocket (see [WebSocket Channels](#websocket-real-time-channels))

---

### Stop Detector Stream
**Endpoint:** `POST /detectors/{alias}/stream/stop`

Stop live detector streaming.

**Request Body:** None

**Response:**
```json
{
  "status": "success",
  "detector_alias": "PRIMARY",
  "streaming": false,
  "timestamp": "2026-01-27T19:34:00Z"
}
```

---

### Capture Single Frame
**Endpoint:** `POST /detectors/{alias}/capture`

Capture a single measurement from detector (synchronous).

**Request Body:**
```json
{
  "frames": 10,           // Number of frames to integrate
  "integration_time": 1.0, // seconds per frame
  "filename_base": "test_capture",  // Optional, auto-generated if not provided
  "save_to_folder": "/path/to/measurements"  // Optional, uses config default if not provided
}
```

**Response:**
```json
{
  "status": "success",
  "detector_alias": "PRIMARY",
  "file_path": "/path/to/measurements/test_capture_PRIMARY.txt",
  "frames_captured": 10,
  "integration_time": 1.0,
  "total_time": 10.234,  // seconds
  "file_size_bytes": 524288,
  "timestamp": "2026-01-27T19:35:00Z"
}
```

---

## Zone Measurements

Zone measurements involve automated scanning of multiple predefined points on a sample.

### Generate Zone Points
**Endpoint:** `POST /measurements/zone/generate-points`

Generate measurement points from zone definitions (server-side farthest-point sampling).

**Request Body:**
```json
{
  "include_zone": {
    "type": "circle",  // or "rectangle"
    "center": {"x": 128, "y": 128},  // pixels
    "radius": 80  // pixels (for circle) or "width"/"height" for rectangle
  },
  "exclude_zones": [  // Optional
    {
      "type": "circle",
      "center": {"x": 100, "y": 100},
      "radius": 20
    }
  ],
  "n_points": 25,
  "shrink_percent": 5,  // Shrink zone by 5%
  "pixel_to_mm_ratio": 10.5,  // pixels per mm
  "include_center_mm": {"x": 0.0, "y": 0.0}  // Real-world coordinates of zone center
}
```

**Response:**
```json
{
  "status": "success",
  "points": [
    {
      "index": 0,
      "pixel_coords": {"x": 128.5, "y": 128.2},
      "mm_coords": {"x": 0.048, "y": 0.019},
      "type": "generated"
    },
    {
      "index": 1,
      "pixel_coords": {"x": 145.3, "y": 132.1},
      "mm_coords": {"x": 1.648, "y": 0.371},
      "type": "generated"
    }
    // ... 23 more points
  ],
  "zones": {
    "include": {
      "type": "circle",
      "center": {"x": 128, "y": 128},
      "radius": 80
    },
    "exclude": [
      {
        "type": "circle",
        "center": {"x": 100, "y": 100},
        "radius": 20
      }
    ]
  },
  "ideal_radius_px": 12.5,  // Suggested zone radius for visualization
  "validation": {
    "all_valid": true,
    "out_of_bounds_count": 0
  },
  "timestamp": "2026-01-27T19:36:00Z"
}
```

---

### Validate Measurement Points
**Endpoint:** `POST /measurements/zone/validate-points`

Validate that all measurement points are within stage limits and not in exclusion zones.

**Request Body:**
```json
{
  "points": [
    {"x": 5.5, "y": -3.2},
    {"x": 6.0, "y": -3.0}
  ],
  "exclude_zones": [  // Optional, in mm coordinates
    {
      "type": "circle",
      "center": {"x": 5.0, "y": -3.0},
      "radius": 0.2
    }
  ],
  "check_spacing": true,  // Optional, default: true
  "min_spacing_mm": 0.1   // Optional
}
```

**Response:**
```json
{
  "all_valid": false,
  "validation_results": [
    {
      "point": {"x": 5.5, "y": -3.2},
      "valid": true,
      "checks": {
        "within_limits": true,
        "not_in_exclusion": true,
        "adequate_spacing": true
      }
    },
    {
      "point": {"x": 6.0, "y": -3.0},
      "valid": false,
      "checks": {
        "within_limits": true,
        "not_in_exclusion": false,
        "adequate_spacing": true
      },
      "errors": ["Point is within exclusion zone at (5.0, -3.0)"]
    }
  ],
  "summary": {
    "total_points": 2,
    "valid_count": 1,
    "invalid_count": 1
  }
}
```

---

### Start Zone Measurement
**Endpoint:** `POST /measurements/zone/start`

Start an automated zone measurement sequence.

**Request Body:**
```json
{
  "measurement_name": "sample_P1_zone1",
  "save_folder": "/path/to/measurements",  // Optional, uses config default
  "sample_image_path": "/path/to/sample_image.jpg",  // Will be embedded in session file
  "points": [
    {
      "index": 0,
      "x": 5.5,
      "y": -3.2,
      "type": "generated"
    },
    {
      "index": 1,
      "x": 6.0,
      "y": -3.0,
      "type": "generated"
    }
  ],
  "zones": {  // Original zone definitions for reference
    "include": {
      "type": "circle",
      "center": {"x": 128, "y": 128},
      "radius": 80
    },
    "exclude": []
  },
  "detector_params": {
    "PRIMARY": {
      "frames": 10,
      "integration_time": 1.0  // seconds
    },
    "SECONDARY": {
      "frames": 10,
      "integration_time": 1.0
    }
  },
  "attenuation_enabled": false,  // Optional
  "point_sorting": "spiral",  // Optional: "spiral", "serpentine", "nearest", default: "spiral"
  "metadata": {  // Optional custom metadata
    "sample_id": "P1",
    "operator": "John Doe",
    "notes": "First measurement of zone 1"
  }
}
```

**Response:**
```json
{
  "status": "started",
  "measurement_id": "meas_20260127_193700_abc123",
  "session_file": "/path/to/measurements/sample_P1_zone1_state.json",
  "total_points": 2,
  "estimated_time_seconds": 45.5,
  "websocket_channel": "measurement_progress",
  "message": "Connect to WebSocket for real-time updates",
  "timestamp": "2026-01-27T19:37:00Z"
}
```

**Errors:**
- `400` - Invalid points or parameters
- `409` - Measurement already active
- `500` - Hardware not initialized

---

### Pause Zone Measurement
**Endpoint:** `POST /measurements/zone/pause`

Pause the current zone measurement (will finish current point then pause).

**Request Body:**
```json
{
  "measurement_id": "meas_20260127_193700_abc123"
}
```

**Response:**
```json
{
  "status": "paused",
  "measurement_id": "meas_20260127_193700_abc123",
  "paused_at_point": 5,
  "total_points": 25,
  "timestamp": "2026-01-27T19:38:00Z"
}
```

---

### Resume Zone Measurement
**Endpoint:** `POST /measurements/zone/resume`

Resume a paused zone measurement.

**Request Body:**
```json
{
  "measurement_id": "meas_20260127_193700_abc123"
}
```

**Response:**
```json
{
  "status": "resumed",
  "measurement_id": "meas_20260127_193700_abc123",
  "resuming_from_point": 6,
  "remaining_points": 19,
  "timestamp": "2026-01-27T19:39:00Z"
}
```

---

### Stop Zone Measurement
**Endpoint:** `POST /measurements/zone/stop`

Stop the current zone measurement (graceful stop).

**Request Body:**
```json
{
  "measurement_id": "meas_20260127_193700_abc123",
  "save_partial_results": true  // Optional, default: true
}
```

**Response:**
```json
{
  "status": "stopped",
  "measurement_id": "meas_20260127_193700_abc123",
  "completed_points": 5,
  "total_points": 25,
  "partial_results_saved": true,
  "session_file": "/path/to/measurements/sample_P1_zone1_state.json",
  "timestamp": "2026-01-27T19:40:00Z"
}
```

---

### Skip Current Point
**Endpoint:** `POST /measurements/zone/skip-point`

Skip the current measurement point and move to next.

**Request Body:**
```json
{
  "measurement_id": "meas_20260127_193700_abc123",
  "reason": "Sample defect at this location"  // Optional
}
```

**Response:**
```json
{
  "status": "point_skipped",
  "skipped_point_index": 5,
  "next_point_index": 6,
  "timestamp": "2026-01-27T19:41:00Z"
}
```

---

### Get Zone Measurement Status
**Endpoint:** `GET /measurements/zone/status`

Get current status of active zone measurement.

**Query Parameters:**
- `measurement_id` (optional): Specific measurement ID, or returns current active measurement

**Response:**
```json
{
  "active": true,
  "measurement_id": "meas_20260127_193700_abc123",
  "status": "running",  // "running", "paused", "stopped", "completed", "error"
  "current_point": 5,
  "total_points": 25,
  "progress_percent": 20.0,
  "completed_points": 5,
  "skipped_points": 0,
  "failed_points": 0,
  "elapsed_time_seconds": 125.3,
  "estimated_remaining_seconds": 476.2,
  "current_stage_position": {"x": 5.5, "y": -3.2},
  "timestamp": "2026-01-27T19:42:00Z"
}
```

---

## Technical Measurements

Technical measurements are single-point measurements with live detector preview.

### Start Technical Measurement
**Endpoint:** `POST /measurements/technical/start`

Perform a single-point measurement at current or specified position.

**Request Body:**
```json
{
  "measurement_name": "alignment_test_001",
  "save_folder": "/path/to/measurements",  // Optional
  "position": {  // Optional, uses current position if not specified
    "x": 5.5,
    "y": -3.2
  },
  "detector_params": {
    "PRIMARY": {
      "frames": 10,
      "integration_time": 1.0
    }
  },
  "move_stage": true,  // Optional, default: false (use current position)
  "metadata": {
    "purpose": "beam alignment check",
    "operator": "Jane Smith"
  }
}
```

**Response:**
```json
{
  "status": "started",
  "measurement_id": "tech_20260127_194300_xyz789",
  "position": {"x": 5.5, "y": -3.2},
  "estimated_time_seconds": 10.5,
  "websocket_channel": "measurement_progress",
  "timestamp": "2026-01-27T19:43:00Z"
}
```

**Errors:**
- `409` - Another measurement is active
- `400` - Invalid parameters
- `500` - Hardware not initialized

---

### Get Technical Measurement Status
**Endpoint:** `GET /measurements/technical/status`

Get status of active technical measurement.

**Response:**
```json
{
  "active": true,
  "measurement_id": "tech_20260127_194300_xyz789",
  "status": "capturing",  // "moving", "capturing", "completed", "error"
  "position": {"x": 5.5, "y": -3.2},
  "progress_percent": 60.0,
  "elapsed_time_seconds": 6.3,
  "estimated_remaining_seconds": 4.2,
  "timestamp": "2026-01-27T19:43:06Z"
}
```

---

### Stop Technical Measurement
**Endpoint:** `POST /measurements/technical/stop`

Stop active technical measurement.

**Request Body:**
```json
{
  "measurement_id": "tech_20260127_194300_xyz789"
}
```

**Response:**
```json
{
  "status": "stopped",
  "measurement_id": "tech_20260127_194300_xyz789",
  "partial_results_saved": true,
  "timestamp": "2026-01-27T19:43:30Z"
}
```

---

## WebSocket Real-time Channels

Connect to: `ws://<server-ip>:8000/api/v1/ws/{channel}`

### Available Channels

#### 1. Stage Position Updates
**Channel:** `ws://server:8000/api/v1/ws/stage_position`

Receive real-time stage position updates (~1 Hz).

**Message Format:**
```json
{
  "type": "stage_position",
  "position": {"x": 5.501, "y": -3.199},
  "moving": true,
  "target": {"x": 6.0, "y": -3.0},  // Only if moving
  "timestamp": "2026-01-27T19:44:00.123Z"
}
```

---

#### 2. Detector Stream
**Channel:** `ws://server:8000/api/v1/ws/detector_stream`

Receive live detector frames (started via `/detectors/{alias}/stream/start`).

**Message Format:**
```json
{
  "type": "detector_frame",
  "detector_alias": "PRIMARY",
  "frame_number": 123,
  "exposure": 0.1,
  "data_format": "base64",  // or "array" for JSON array
  "data": "base64_encoded_frame_data...",
  // Alternative: "data": [[pixel_values...], [...], ...] for "array" format
  "shape": [256, 256],
  "timestamp": "2026-01-27T19:44:01.234Z"
}
```

**Note:** For high-frequency updates, base64 encoding is more efficient. Client decodes to Float32Array or similar.

---

#### 3. Measurement Progress
**Channel:** `ws://server:8000/api/v1/ws/measurement_progress`

Receive real-time measurement progress updates.

**Message Types:**

**Started:**
```json
{
  "type": "measurement_started",
  "measurement_id": "meas_20260127_193700_abc123",
  "measurement_type": "zone",  // or "technical"
  "total_points": 25,
  "timestamp": "2026-01-27T19:45:00.000Z"
}
```

**Progress:**
```json
{
  "type": "measurement_progress",
  "measurement_id": "meas_20260127_193700_abc123",
  "current_point": 5,
  "total_points": 25,
  "progress_percent": 20.0,
  "current_position": {"x": 5.5, "y": -3.2},
  "current_detector": "PRIMARY",
  "elapsed_time_seconds": 125.3,
  "estimated_remaining_seconds": 476.2,
  "timestamp": "2026-01-27T19:45:05.123Z"
}
```

**Point Completed:**
```json
{
  "type": "point_completed",
  "measurement_id": "meas_20260127_193700_abc123",
  "point_index": 5,
  "position": {"x": 5.5, "y": -3.2},
  "files": [
    "/path/to/measurements/sample_P1_zone1_5.50_-3.20_20260127_194505_PRIMARY.txt",
    "/path/to/measurements/sample_P1_zone1_5.50_-3.20_20260127_194505_SECONDARY.txt"
  ],
  "timestamp": "2026-01-27T19:45:15.456Z"
}
```

**Completed:**
```json
{
  "type": "measurement_completed",
  "measurement_id": "meas_20260127_193700_abc123",
  "total_points": 25,
  "completed_points": 25,
  "skipped_points": 0,
  "failed_points": 0,
  "total_time_seconds": 625.7,
  "session_file": "/path/to/measurements/sample_P1_zone1_state.json",
  "timestamp": "2026-01-27T19:55:25.789Z"
}
```

**Error:**
```json
{
  "type": "measurement_error",
  "measurement_id": "meas_20260127_193700_abc123",
  "error_type": "stage_timeout",  // or "detector_failure", "connection_lost", etc.
  "error_message": "Stage movement timed out at point 5",
  "point_index": 5,
  "position": {"x": 5.5, "y": -3.2},
  "recovery_action": "paused",  // "paused", "aborted", "retrying"
  "timestamp": "2026-01-27T19:46:00.000Z"
}
```

---

#### 4. Hardware Status
**Channel:** `ws://server:8000/api/v1/ws/hardware_status`

Receive hardware connection/status change notifications.

**Message Format:**
```json
{
  "type": "hardware_status_change",
  "component": "stage",  // or "detector"
  "component_id": "XY_STAGE",  // or detector alias
  "status": "disconnected",  // "connected", "disconnected", "error"
  "message": "Stage communication timeout",
  "timestamp": "2026-01-27T19:47:00.000Z"
}
```

---

#### 5. Measurement Log/Events
**Channel:** `ws://server:8000/api/v1/ws/events`

Receive real-time log events and system notifications.

**Message Format:**
```json
{
  "type": "log_event",
  "level": "warning",  // "debug", "info", "warning", "error", "critical"
  "category": "measurement",  // "hardware", "measurement", "system", "config"
  "message": "Skipping measurement point 5 at (5.5, -3.2) mm - outside limits",
  "details": {
    "point_index": 5,
    "position": {"x": 5.5, "y": -3.2},
    "reason": "axis_limit_exceeded"
  },
  "timestamp": "2026-01-27T19:48:00.000Z"
}
```

---

## File Management

### List Measurement Files
**Endpoint:** `GET /files/measurements`

List measurement files in a directory.

**Query Parameters:**
- `folder`: Path to folder (optional, uses config default)
- `pattern`: File pattern filter (optional, e.g., "*.txt")
- `recursive`: Boolean (optional, default: false)

**Response:**
```json
{
  "folder": "/path/to/measurements",
  "files": [
    {
      "name": "sample_P1_zone1_5.50_-3.20_20260127_194505_PRIMARY.txt",
      "path": "/path/to/measurements/sample_P1_zone1_5.50_-3.20_20260127_194505_PRIMARY.txt",
      "size_bytes": 524288,
      "created": "2026-01-27T19:45:15Z",
      "type": "detector_data"
    },
    {
      "name": "sample_P1_zone1_state.json",
      "path": "/path/to/measurements/sample_P1_zone1_state.json",
      "size_bytes": 12345,
      "created": "2026-01-27T19:55:25Z",
      "type": "session_file"
    }
  ],
  "total_count": 2,
  "total_size_bytes": 536633
}
```

---

### Download File
**Endpoint:** `GET /files/download`

Download a measurement file.

**Query Parameters:**
- `path`: Full file path

**Response:** File stream (application/octet-stream)

**Headers:**
- `Content-Disposition: attachment; filename="sample_P1_zone1_state.json"`
- `Content-Type: application/octet-stream`

---

### Get Session File
**Endpoint:** `GET /files/session/{measurement_id}`

Retrieve the JSON session file for a measurement.

**Response:**
```json
{
  "measurement_id": "meas_20260127_193700_abc123",
  "measurement_name": "sample_P1_zone1",
  "measurement_type": "zone",
  "start_time": "2026-01-27T19:37:00Z",
  "end_time": "2026-01-27T19:55:25Z",
  "total_time_seconds": 625.7,
  "image_base64": "data:image/jpeg;base64,/9j/4AAQSkZJRg...",
  "points": [...],
  "zones": {...},
  "measurement_points": [...],
  "skipped_points": [],
  "detector_params": {...},
  "stage_config": {...},
  "poni_files": {...},
  "metadata": {...},
  "CALIBRATION_GROUP_HASH": "a1b2c3d4e5f6g7h8"
}
```

---

### Upload to Matador Cloud
**Endpoint:** `POST /files/upload-to-matador`

**Note:** This endpoint is a placeholder for v1.0. The server will contact Matador cloud API and upload measurement data.

**Request Body:**
```json
{
  "measurement_id": "meas_20260127_193700_abc123",
  "matador_endpoint": "https://matador.cloud/api/v1/upload",  // Optional, uses config default
  "include_raw_files": true  // Optional, default: true
}
```

**Response:**
```json
{
  "status": "success",
  "message": "Data uploaded to Matador cloud",
  "matador_measurement_id": "matador_xyz789abc",
  "upload_time_seconds": 45.3,
  "uploaded_files_count": 52,
  "timestamp": "2026-01-27T19:56:00Z"
}
```

**Status:** To be implemented - interface defined for future integration.

---

## Error Handling

### Error Response Format

All error responses follow this format:

```json
{
  "error": true,
  "error_type": "hardware_error",  // Category of error
  "message": "Stage movement timed out",  // Human-readable message
  "details": {  // Optional additional context
    "component": "stage",
    "operation": "move",
    "position": {"x": 5.5, "y": -3.2}
  },
  "recovery_suggestions": [  // Optional recovery steps
    "Check stage power and connections",
    "Try re-initializing hardware",
    "Contact support if problem persists"
  ],
  "timestamp": "2026-01-27T19:49:00Z"
}
```

### Error Types

| Error Type | HTTP Code | Description | Recovery Action |
|------------|-----------|-------------|-----------------|
| `hardware_not_initialized` | 500 | Hardware not initialized | Call `/hardware/initialize` |
| `hardware_connection_lost` | 500 | Lost connection to hardware | Pause measurement, manual intervention |
| `stage_timeout` | 500 | Stage movement timeout | Pause measurement, check hardware |
| `stage_limit_exceeded` | 400 | Coordinates outside limits | Validate coordinates before submission |
| `detector_capture_failed` | 500 | Detector capture error | Abort measurement (per requirement) |
| `measurement_active` | 409 | Another measurement running | Wait or stop active measurement |
| `invalid_parameters` | 400 | Invalid request parameters | Check request body format |
| `file_not_found` | 404 | Requested file doesn't exist | Verify file path |
| `config_error` | 500 | Configuration file issue | Check config file validity |
| `setup_not_found` | 404 | Setup config file not found | List available setups |

### Error Handling Strategy (Per Requirements)

1. **Hardware Connection Errors:** Pause measurement, notify client via WebSocket, wait for manual recovery
2. **Stage Timeout/Collision:** Pause measurement, notify client, allow retry/skip/abort decision
3. **Detector Failure:** Abort measurement immediately (conservative approach)
4. **General Errors:** Return detailed error response with recovery suggestions

---

## Data Models

### Common Types

```python
# Coordinate (mm)
{
  "x": float,
  "y": float
}

# Pixel Coordinate
{
  "x": int,
  "y": int
}

# Zone Definition
{
  "type": "circle" | "rectangle",
  "center": PixelCoordinate,
  "radius": float,  # for circle
  "width": float,   # for rectangle
  "height": float   # for rectangle
}

# Measurement Point
{
  "index": int,
  "x": float,  # mm
  "y": float,  # mm
  "type": "generated" | "user",
  "pixel_coords": PixelCoordinate  # optional, for reference
}

# Detector Parameters
{
  "frames": int,
  "integration_time": float  # seconds
}

# Stage Configuration
{
  "alias": str,
  "type": "Kinesis" | "Marlin" | "DummyStage",
  "limits": {
    "x": [float, float],  # [min, max]
    "y": [float, float]
  },
  "home_position": [float, float],
  "load_position": [float, float]
}

# Detector Configuration
{
  "alias": str,
  "type": "Pixet" | "DummyDetector",
  "id": str,
  "size": {"width": int, "height": int},
  "active": bool,
  "poni_file": str  # path to calibration file
}
```

---

## Implementation Notes

### Client-Side Recommendations

1. **WebSocket Connection Management:**
   - Establish WebSocket connections after starting operations that need real-time updates
   - Handle reconnection logic for dropped connections
   - Close connections when no longer needed

2. **Image Handling:**
   - Client handles all interactive drawing and annotation
   - Upload final sample image with measurement request
   - Server embeds image in session JSON file (base64)

3. **Zone Point Generation:**
   - Send zone definitions to server for point generation
   - Server returns validated points
   - Client visualizes points on image before starting measurement

4. **Error Handling:**
   - Subscribe to WebSocket events channel for error notifications
   - Implement pause/recovery UI for hardware errors
   - Provide manual intervention options (retry, skip, abort)

5. **File Access:**
   - Use real-time file access endpoints during measurements
   - Download session JSON after measurement completion
   - Handle large file downloads efficiently (streaming)

### Server-Side Implementation Notes

1. **Hardware Abstraction:**
   - Use existing `HardwareController` class
   - Wrap in async handlers for FastAPI
   - Maintain single global hardware instance (mutex for measurement operations)

2. **WebSocket Broadcasting:**
   - Use FastAPI WebSocket manager for multi-client support
   - Broadcast updates to all connected clients on respective channels
   - Handle client disconnections gracefully

3. **File System:**
   - Store measurement files in configured folder structure
   - Generate unique measurement IDs (timestamp + hash)
   - Periodic cleanup not implemented (manual only, per requirement)

4. **Configuration Hot-Swapping:**
   - Load config files dynamically
   - Reinitialize hardware when config changes affect hardware
   - Validate config before applying

5. **Concurrency:**
   - Only one measurement (zone OR technical) can run at a time
   - Use asyncio locks/semaphores for hardware access
   - Queue mechanism not needed (single active measurement)

---

## Version History

**v1.0.0** (2026-01-27)
- Initial API specification
- FastAPI framework
- No authentication
- Manual hardware initialization
- Full real-time WebSocket support
- File-based storage
- Zone and Technical measurement workflows
- Conservative error handling with manual recovery
- Matador cloud integration interface defined (implementation TBD)

---

## Future Considerations (Beyond v1.0)

1. **Authentication & Authorization** - Add API keys or JWT tokens
2. **Database Integration** - Store metadata for easier querying
3. **Multi-user Coordination** - Queue system for multiple operators
4. **Advanced Analytics** - Real-time data processing endpoints
5. **Matador Cloud Integration** - Complete upload automation
6. **Hardware Monitoring** - Predictive maintenance alerts
7. **Measurement Templates** - Save and reuse common configurations
8. **Batch Operations** - Queue multiple measurement jobs

---

**End of API v1.0 Documentation**
