# DiFRA API v1.0 - Summary & Quick Reference

**Date:** 2026-01-27  
**Status:** Specification Complete - Ready for Implementation

---

## Key Decisions Summary

### Architecture
| Decision | Choice | Rationale |
|----------|--------|-----------|
| Framework | **FastAPI** | Async, WebSocket, auto-docs, type hints |
| Authentication | **None (v1.0)** | Focus on functionality first |
| Hardware Init | **Manual per session** | Full control, matches PyQt5 app |
| Real-time | **WebSocket (5 channels)** | All updates via WS |
| Storage | **File-based** | Data flows to Matador cloud |
| Image Handling | **Client UI + server storage** | Client draws, server embeds in JSON |
| Measurement Mutex | **Zone OR Technical** | Only one type runs at a time |

### Hardware Control
- **Stage:** Manual init/deinit, emergency stop, full validation (limits + exclusions + spacing)
- **Detectors:** Config-driven with API rewrite capability, per-measurement parameters
- **PONI Files:** Upload/manage via API, flexible loading locations

### Error Handling
- **Hardware errors:** Pause → manual recovery (if possible), else abort
- **Stage timeout:** Pause → allow retry/skip/abort
- **Detector failure:** **Abort immediately** (conservative)

---

## API Endpoints Overview

### Hardware Management (4 endpoints)
```
POST   /hardware/initialize        # Init all hardware
POST   /hardware/deinitialize      # Deinit all hardware  
POST   /hardware/reinitialize      # Recovery reinit
GET    /hardware/status            # Current status
```

### Configuration Management (7 endpoints)
```
GET    /config                     # Get current config
PUT    /config                     # Update config
GET    /config/setups              # List available setups
POST   /config/setup/switch        # Hot-swap setup (no restart!)
GET    /config/poni                # List PONI files
POST   /config/poni/upload         # Upload PONI file
POST   /config/poni/set            # Set PONI for detector
```

### Stage Control (6 endpoints)
```
POST   /stage/move                 # Move to position
GET    /stage/position             # Get current position
POST   /stage/home                 # Home stage
POST   /stage/load                 # Move to load position
POST   /stage/emergency-stop       # Emergency halt
POST   /stage/validate             # Pre-validate coordinates
```

### Detector Control (4 endpoints)
```
GET    /detectors                  # List detectors
POST   /detectors/{alias}/stream/start   # Start live stream
POST   /detectors/{alias}/stream/stop    # Stop stream
POST   /detectors/{alias}/capture        # Single capture
```

### Zone Measurements (7 endpoints)
```
POST   /measurements/zone/generate-points   # Server-side point generation
POST   /measurements/zone/validate-points   # Full validation
POST   /measurements/zone/start             # Start measurement
POST   /measurements/zone/pause             # Pause
POST   /measurements/zone/resume            # Resume
POST   /measurements/zone/stop              # Stop (graceful)
POST   /measurements/zone/skip-point        # Skip current point
GET    /measurements/zone/status            # Get status
```

### Technical Measurements (3 endpoints)
```
POST   /measurements/technical/start   # Single-point measurement
GET    /measurements/technical/status  # Get status
POST   /measurements/technical/stop    # Stop
```

### File Management (4 endpoints)
```
GET    /files/measurements            # List files
GET    /files/download                # Download file
GET    /files/session/{id}            # Get session JSON
POST   /files/upload-to-matador       # Upload to cloud (TBD)
```

**Total: 35 REST endpoints**

---

## WebSocket Channels (5 channels)

```
ws://<server>:8000/api/v1/ws/stage_position       # Real-time XY position (~1 Hz)
ws://<server>:8000/api/v1/ws/detector_stream      # Live detector frames
ws://<server>:8000/api/v1/ws/measurement_progress # Progress updates
ws://<server>:8000/api/v1/ws/hardware_status      # Connection status changes
ws://<server>:8000/api/v1/ws/events               # Log events & notifications
```

---

## Implementation Priorities

### Phase 1: Core Infrastructure (Week 1-2)
1. **FastAPI server setup**
   - Project structure
   - Async wrappers for `HardwareController`
   - Configuration loading
   
2. **Hardware Management**
   - Initialize/deinitialize endpoints
   - Status monitoring
   - Error handling framework

3. **WebSocket Foundation**
   - Connection manager
   - Broadcasting infrastructure
   - Basic status updates

### Phase 2: Motion & Detection (Week 3-4)
4. **Stage Control**
   - Movement endpoints
   - Validation logic
   - Emergency stop

5. **Detector Control**
   - Capture endpoints
   - Streaming setup
   - Multi-detector coordination

6. **Configuration Management**
   - CRUD operations
   - Hot-swapping
   - PONI file management

### Phase 3: Measurements (Week 5-7)
7. **Zone Measurements**
   - Point generation algorithm
   - Measurement orchestration
   - Pause/resume/skip logic
   - Progress tracking

8. **Technical Measurements**
   - Single-point workflow
   - Live preview integration

9. **File Management**
   - File listing/download
   - Session file generation

### Phase 4: Real-time & Polish (Week 8)
10. **WebSocket Completion**
    - All 5 channels fully implemented
    - Detector frame streaming
    - Progress updates
    - Error notifications

11. **Testing & Documentation**
    - Integration tests
    - API client examples
    - Deployment guide

---

## Key Implementation Notes

### Hardware Abstraction
```python
# Wrap existing HardwareController in async
from hardware.difra.hardware.hardware_control import HardwareController

class AsyncHardwareManager:
    def __init__(self):
        self.hw_controller = None
        self.lock = asyncio.Lock()
    
    async def initialize(self, config):
        async with self.lock:
            self.hw_controller = HardwareController(config)
            return await asyncio.to_thread(self.hw_controller.initialize)
```

### Measurement Mutex
```python
# Global measurement state
class MeasurementManager:
    def __init__(self):
        self.active_measurement_id = None
        self.measurement_type = None  # "zone" or "technical"
        self.lock = asyncio.Lock()
    
    async def start_measurement(self, type):
        async with self.lock:
            if self.active_measurement_id:
                raise HTTPException(409, "Measurement already active")
            self.active_measurement_id = generate_id()
            self.measurement_type = type
```

### WebSocket Broadcasting
```python
# Connection manager
class ConnectionManager:
    def __init__(self):
        self.active_connections = {
            "stage_position": [],
            "detector_stream": [],
            "measurement_progress": [],
            "hardware_status": [],
            "events": []
        }
    
    async def broadcast(self, channel, message):
        for connection in self.active_connections[channel]:
            await connection.send_json(message)
```

---

## File Structure Recommendation

```
omniscan-hw-server/
├── app/
│   ├── __init__.py
│   ├── main.py                    # FastAPI app entry
│   ├── config.py                  # Config management
│   ├── dependencies.py            # Shared dependencies
│   │
│   ├── api/
│   │   ├── __init__.py
│   │   ├── v1/
│   │   │   ├── __init__.py
│   │   │   ├── hardware.py        # Hardware endpoints
│   │   │   ├── config.py          # Config endpoints
│   │   │   ├── stage.py           # Stage endpoints
│   │   │   ├── detectors.py       # Detector endpoints
│   │   │   ├── zone_measurements.py    # Zone endpoints
│   │   │   ├── technical_measurements.py  # Technical endpoints
│   │   │   ├── files.py           # File endpoints
│   │   │   └── websockets.py      # WS endpoints
│   │
│   ├── core/
│   │   ├── __init__.py
│   │   ├── hardware_manager.py    # Async hardware wrapper
│   │   ├── measurement_manager.py # Measurement orchestration
│   │   ├── websocket_manager.py   # WS connection manager
│   │   └── point_generator.py     # Zone point generation
│   │
│   ├── models/
│   │   ├── __init__.py
│   │   ├── hardware.py            # Pydantic models
│   │   ├── measurements.py
│   │   ├── config.py
│   │   └── responses.py
│   │
│   └── utils/
│       ├── __init__.py
│       ├── validation.py          # Coordinate validation
│       └── errors.py              # Custom exceptions
│
├── tests/
│   ├── test_hardware.py
│   ├── test_measurements.py
│   └── test_websockets.py
│
├── requirements.txt
├── Dockerfile
└── README.md
```

---

## Next Steps

### Immediate (This Week)
1. **Create server repository** (`omniscan-hw-server`)
2. **Set up FastAPI project structure**
3. **Implement hardware management endpoints** (init/deinit/status)
4. **Create basic WebSocket infrastructure**

### Short-term (Next 2 Weeks)
5. **Implement stage control endpoints**
6. **Implement detector control endpoints**
7. **Add configuration management**
8. **Set up testing framework**

### Medium-term (Next Month)
9. **Implement zone measurements**
10. **Implement technical measurements**
11. **Complete all WebSocket channels**
12. **Integration testing with PyQt5 client**

### Long-term (Next Quarter)
13. **Build web UI client** (framework TBD by colleagues)
14. **Matador cloud integration**
15. **Performance optimization**
16. **Production deployment**

---

## Testing Strategy

### Unit Tests
- Hardware controller wrappers
- Point generation algorithm
- Validation logic
- Configuration management

### Integration Tests
- Hardware initialization flows
- Measurement workflows (zone & technical)
- WebSocket message broadcasting
- Error handling & recovery

### End-to-End Tests
- Complete measurement sequences
- Multi-client WebSocket scenarios
- File generation & download
- Config hot-swapping

---

## Dependencies

### Core
```
fastapi==0.109.0
uvicorn[standard]==0.27.0
websockets==12.0
pydantic==2.5.0
python-multipart==0.0.6  # For file uploads
```

### Existing Hardware Libraries
```
# Already in xrd-analysis
pyserial>=3.5  # For Marlin stages
numpy>=1.24.0
```

### Optional
```
aiofiles==23.2.1  # Async file operations
python-jose[cryptography]==3.3.0  # For future JWT auth
```

---

## Configuration Example

```json
{
  "server": {
    "host": "0.0.0.0",
    "port": 8000,
    "workers": 1,
    "reload": false
  },
  "hardware": {
    "setup_config_path": "/path/to/xrd-analysis/src/hardware/difra/resources/config",
    "default_setup": "Ulster (Xena)",
    "auto_init_on_startup": false
  },
  "measurements": {
    "default_folder": "/data/measurements",
    "allow_concurrent": false,
    "max_history": 100
  },
  "matador": {
    "endpoint": "https://matador.cloud/api/v1/upload",
    "enabled": false
  }
}
```

---

## Documentation Files Created

1. **`API_v1.0_ENDPOINTS.md`** (1579 lines)
   - Complete endpoint documentation
   - Request/response examples
   - WebSocket message formats
   - Error handling patterns
   - Implementation notes

2. **`API_v1.0_SUMMARY.md`** (This file)
   - Quick reference
   - Implementation priorities
   - File structure
   - Next steps

---

## Questions/Decisions Deferred

### For Future Discussion
1. **Matador Cloud Integration Details**
   - Authentication mechanism
   - Upload format/protocol
   - Retry logic
   - Rate limiting

2. **Web UI Framework**
   - Will be decided by your colleagues
   - API is framework-agnostic

3. **Multi-Setup Support**
   - Currently supports hot-swapping between setups
   - Future: Run multiple setups simultaneously?

4. **Advanced Features**
   - Measurement templates/presets
   - Historical data analysis
   - Predictive maintenance
   - Batch job queue

---

**Ready for implementation! 🚀**

For questions or clarifications, refer to the full documentation in `API_v1.0_ENDPOINTS.md`.
