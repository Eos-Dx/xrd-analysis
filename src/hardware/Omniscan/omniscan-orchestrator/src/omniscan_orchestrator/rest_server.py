"""
REST API Server for Omniscan Orchestrator

FastAPI server that bridges Web UI (password auth) with Hardware Server (mTLS gRPC).
Implements privacy-by-design: patient PII stays in orchestrator database only.
"""

from fastapi import FastAPI, HTTPException, Depends, Header, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from typing import Optional, Dict, List
from datetime import datetime, timedelta
import uuid
import asyncio
import json
from contextlib import asynccontextmanager

from .models import (
    LoginRequest, LoginResponse, LogoutResponse,
    MeasurementStartRequest, MeasurementStartResponse,
    MeasurementResult, MeasurementHistoryResponse,
    CalibrationStartResponse, CalibrationStatus,
    SystemHealth, SafetyInterlocks, ErrorResponse
)
from .grpc_client import OmniscanGrpcClient
from .database import OrchestratorDatabase, PatientRecord, MeasurementRecord
from .database import CalibrationRecord as DBCalibrationRecord
from .auth_manager import AuthenticationManager
from pathlib import Path
import os


# Global state
grpc_client: Optional[OmniscanGrpcClient] = None
db: Optional[OrchestratorDatabase] = None
auth_manager: Optional[AuthenticationManager] = None
websocket_connections: List[WebSocket] = []

# Active run tracking for measurement heartbeats
active_runs: Dict[str, Dict] = {}  # run_id -> {"type": "measurement|calibration", "started": timestamp, "total_secs": int, "elapsed": int, "status": "running|completed|failed"}


# Background task for monitoring state changes
state_monitor_task: Optional[asyncio.Task] = None

# Calibration background task
active_calibration_task: Optional[asyncio.Task] = None
current_calibration_id: Optional[str] = None

async def monitor_state_changes():
    """Background task to subscribe to hardware server state changes."""
    global grpc_client
    
    print("📡 Starting state change monitor...")
    
    while True:
        try:
            # Subscribe to state updates stream
            print("🔄 Attempting to subscribe to state updates...")
            stream = grpc_client.subscribe_to_state_updates()
            
            if stream is None:
                print("⚠️  Failed to subscribe to state updates, retrying in 5s...")
                await asyncio.sleep(5)
                continue
            
            print("✅ Subscribed to hardware server state updates")
            print("👂 Listening for state changes...")
            
            # Process incoming notifications one at a time
            # Use asyncio.to_thread to avoid blocking the event loop
            while True:
                try:
                    # Get next notification in a thread to avoid blocking
                    notification = await asyncio.to_thread(lambda: next(stream, None))
                    
                    if notification is None:
                        print("⚠️  Stream ended, reconnecting...")
                        break
                    
                    component = notification.component
                    change_type = notification.change_type
                    
                    print(f"📢 State change: {component} - {change_type}")
                    
                    # Handle RUN notifications for measurement/calibration progress
                    if component == "RUN" and change_type.startswith(("START:", "TICK:", "DONE:")):
                        await handle_run_notification(change_type)
                    
                    # Query actual state and broadcast to WebSocket clients
                    event_data = {
                        "type": "state_change",
                        "component": component,
                        "change_type": change_type,
                        "timestamp": notification.timestamp.seconds if notification.timestamp else None
                    }
                    
                    # For specific change types, query and include actual state
                    if change_type == "INTERLOCK_CHANGED":
                        gpio_state = grpc_client.get_gpio_state()
                        if "error" not in gpio_state:
                            event_data["interlocks"] = gpio_state.get("interlocks")
                            print(f"  ℹ️  Interlocks: overall_safe={gpio_state.get('interlocks', {}).get('overall_safe')}")
                    elif change_type == "KEY_SWITCH_CHANGED":
                        gpio_state = grpc_client.get_gpio_state()
                        if "error" not in gpio_state:
                            event_data["key_switch_on"] = gpio_state.get("key_switch_on")
                            print(f"  ℹ️  Key Switch: {gpio_state.get('key_switch_on')}")
                    elif change_type == "ENABLE_BUTTON_ACTIVATED":
                        gpio_state = grpc_client.get_gpio_state()
                        if "error" not in gpio_state:
                            event_data["activation_button_active"] = gpio_state.get("activation_button_active")
                            event_data["activation_remaining_secs"] = gpio_state.get("activation_remaining_secs")
                            print(f"  ℹ️  Activation Button: active={gpio_state.get('activation_button_active')}, remaining={gpio_state.get('activation_remaining_secs')}s")
                    elif change_type == "ENABLE_BUTTON_DEACTIVATED":
                        # Button expired - set to False
                        event_data["activation_button_active"] = False
                        event_data["activation_remaining_secs"] = 0
                        print(f"  ℹ️  Activation Button: DEACTIVATED (expired)")
                    
                    # Broadcast to WebSocket clients
                    await broadcast_event(event_data)
                    
                except StopIteration:
                    print("⚠️  Stream ended, reconnecting...")
                    break
                
        except Exception as e:
            print(f"❌ State monitor error: {e}")
            print("Reconnecting in 5 seconds...")
            await asyncio.sleep(5)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup and shutdown."""
    global grpc_client, db, auth_manager, state_monitor_task

    # Test hook: allow disabling startup side effects so pytest patches work
    if os.getenv("OMNI_TEST_NO_LIFESPAN") == "1":
        yield
        return
    
    # Startup: Initialize connections
    print("🚀 Starting Omniscan Orchestrator REST API...")
    
    # Initialize database - resolve path relative to project root
    # Get project root (3 levels up from rest_server.py location)
    project_root = Path(__file__).parent.parent.parent
    db_path = project_root / "data" / "orchestrator.db"
    db = OrchestratorDatabase(str(db_path))
    print(f"✅ Database initialized at {db_path}")
    
    # Initialize gRPC client with database for command logging (configure with certs in production)
    grpc_client = OmniscanGrpcClient(
        server_address="localhost:50051",
        database=db,  # Pass database for command logging
        session_id=str(uuid.uuid4()),  # Session ID for this orchestrator instance
        # TODO: Load from config file
        # client_cert_path="certs/orchestrator.crt",
        # client_key_path="certs/orchestrator.key",
        # ca_cert_path="certs/ca.crt"
    )
    print("✅ gRPC client initialized with command logging")
    
    # Validate compatibility with hardware server
    print("🔍 Checking compatibility with hardware server...")
    try:
        compat = grpc_client.validate_compatibility(
            client_version="0.1.0",
            protocol_version="1.0.0"
        )
        
        if "error" in compat:
            print(f"⚠️  Could not validate compatibility: {compat['error']}")
            print("⚠️  Proceeding without compatibility check (server may not support CommandDiscovery)")
        elif not compat["compatible"]:
            print(f"❌ INCOMPATIBILITY DETECTED: {compat['message']}")
            if compat["missing_commands"]:
                print(f"   Missing commands: {', '.join(compat['missing_commands'])}")
            if compat["version_warnings"]:
                for warning in compat["version_warnings"]:
                    print(f"   Warning: {warning}")
            print("")
            print("❌ I cannot communicate with the server, I am outdated")
            print("❌ Orchestrator startup ABORTED due to incompatibility")
            print("   Please update the orchestrator or hardware server to compatible versions")
            raise RuntimeError("Server incompatibility detected - cannot continue")
        else:
            print(f"✅ Compatibility check passed: {compat['message']}")
            if compat.get("version_warnings"):
                for warning in compat["version_warnings"]:
                    print(f"   ⚠️  {warning}")
    except RuntimeError:
        raise  # Re-raise compatibility errors
    except Exception as e:
        print(f"⚠️  Compatibility check failed with exception: {e}")
        print("⚠️  Proceeding anyway (server may not support CommandDiscovery yet)")
    
    # Initialize authentication manager
    auth_manager = AuthenticationManager(grpc_client, db)
    print("✅ Authentication manager initialized")
    
    # Start state monitoring background task
    state_monitor_task = asyncio.create_task(monitor_state_changes())
    print("✅ State monitor background task started")
    
    print("🌐 REST API ready (check Uvicorn log for actual port)")
    
    yield
    
    # Shutdown: Clean up connections
    print("🛑 Shutting down Omniscan Orchestrator REST API...")
    
    # Cancel state monitor task
    if state_monitor_task:
        state_monitor_task.cancel()
        try:
            await state_monitor_task
        except asyncio.CancelledError:
            pass
    
    if grpc_client:
        grpc_client.close()
    print("✅ Cleanup complete")


# Create FastAPI app
app = FastAPI(
    title="Omniscan Orchestrator API",
    description="Medical device orchestration API for Omniscan XRD system",
    version="0.1.0",
    lifespan=lifespan
)

# CORS middleware for Web UI
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",  # React dev server
        "http://localhost:3001",  # Alternate React port
        "http://127.0.0.1:3000",
        "http://127.0.0.1:3001",
        "*"  # Allow all origins in development (remove in production)
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================================
# Authentication Dependencies
# ============================================================================

async def get_current_user(x_session_id: Optional[str] = Header(None)) -> Dict[str, str]:
    """Dependency to extract and validate session from header."""
    if not x_session_id:
        raise HTTPException(status_code=401, detail="Missing session ID")
    
    if not auth_manager.is_session_valid(x_session_id):
        raise HTTPException(status_code=401, detail="Invalid or expired session")
    
    # Update activity timestamp
    auth_manager.update_activity(x_session_id)
    
    session = auth_manager.get_session(x_session_id)
    return {
        "user_id": session.user_id,
        "role": session.role,
        "session_id": x_session_id
    }


# ============================================================================
# Authentication Endpoints
# ============================================================================

@app.post("/api/auth/login", response_model=LoginResponse)
async def login(request: LoginRequest):
    """
    Authenticate user with password and key switch verification.
    
    Process:
    1. Verify username/password (local DB or LDAP)
    2. Check hardware key switch is ON (via gRPC)
    3. Create session if both pass
    """
    # TODO: Determine role from user database lookup
    # For now, default to operator
    role = "operator"
    
    result = await auth_manager.login(
        user_id=request.username,
        password=request.password,
        role=role
    )
    
    if result.success:
        return LoginResponse(
            success=True,
            session_id=result.session_id,
            user_id=request.username,
            role=role
        )
    else:
        return LoginResponse(
            success=False,
            error=result.error_message
        )


@app.post("/api/auth/logout", response_model=LogoutResponse)
async def logout(user: Dict = Depends(get_current_user)):
    """Logout user and invalidate session."""
    success = auth_manager.logout(user["session_id"])
    
    if success:
        return LogoutResponse(success=True, message="Logged out successfully")
    else:
        return LogoutResponse(success=False, message="Logout failed")


# ============================================================================
# System Health Endpoints
# ============================================================================

@app.get("/api/state")
async def get_system_state(user: Dict = Depends(get_current_user)):
    """
    Get current system state (for UI polling).
    
    This is a simplified endpoint that returns just the essential state info
    that the UI needs for its state machine and display.
    
    Returns:
    - state: Current server state (IDLE, PENDING_ARMED, RUNNING, etc.)
    - interlocks: Safety interlock status
    - devices: Device status (detector, motion)
    - calibration: Calibration validity
    """
    # Get full state from hardware server
    full_state = grpc_client.get_full_server_state()
    
    if "error" in full_state:
        raise HTTPException(status_code=503, detail=full_state["error"])
    
    # Get GPIO state for interlocks
    gpio_state = grpc_client.get_gpio_state()
    
    # Get individual device states
    pdu_state = grpc_client.get_device_state_general("pdu")
    gpio_device_state = grpc_client.get_device_state_general("gpio")
    
    # Build simplified response
    return {
        "state": full_state.get("safety_state", "UNKNOWN"),
        "timestamp": datetime.utcnow().isoformat(),
        "interlocks": {
            "overall_safe": gpio_state.get("interlocks", {}).get("overall_safe", False),
            "key_switch": gpio_state.get("key_switch_on", False),
            "enable_button": gpio_state.get("activation_button_active", False),
            "door_closed": gpio_state.get("interlocks", {}).get("door_closed", False),
            "emergency_stop": gpio_state.get("interlocks", {}).get("emergency_stop", False),
            "radiation_safe": gpio_state.get("interlocks", {}).get("radiation_safe", False),
            "cooling_ok": gpio_state.get("interlocks", {}).get("cooling_ok", False),
            "power_ok": gpio_state.get("interlocks", {}).get("power_ok", False),
        } if "error" not in gpio_state else None,
        "devices": {
            "pdu": {
                "powered": pdu_state.get("powered", False),
                "status": pdu_state.get("status", "Off"),
            },
            "gpio": {
                "powered": gpio_device_state.get("powered", False),
                "status": gpio_device_state.get("status", "Off"),
            },
            "detector": {
                "powered": full_state.get("detector", {}).get("powered", False),
                "initialized": full_state.get("detector", {}).get("initialized", False),
                "status": full_state.get("detector", {}).get("status", "OFF"),
                "temperature": full_state.get("detector", {}).get("temperature"),
                "total_exposures": full_state.get("detector", {}).get("total_exposures", 0),
            },
            "motion": {
                "powered": full_state.get("motion", {}).get("powered", False),
                "initialized": full_state.get("motion", {}).get("initialized", False),
                "status": full_state.get("motion", {}).get("status", "OFF"),
                "is_homed": full_state.get("motion", {}).get("is_homed", False),
                "position_x": full_state.get("motion", {}).get("position_x"),
                "position_y": full_state.get("motion", {}).get("position_y"),
                "total_moves": full_state.get("motion", {}).get("total_moves", 0),
            },
        },
        "calibration": {
            "valid": True,  # TODO: Get from hardware server
            "expires_at": None,  # TODO: Get from hardware server
        },
    }


@app.get("/api/health", response_model=SystemHealth)
async def get_system_health(user: Dict = Depends(get_current_user)):
    """
    Get overall system health from hardware server.
    
    Returns individual device statuses:
    - PDU: Power distribution unit state
    - GPIO: Always active (manages interlocks)
    - Detector: Current state (OFF, IDLE, INIT, etc.)
    - Motion: Current state (OFF, IDLE, INIT, etc.)
    """
    status = grpc_client.get_status()
    
    if "error" in status:
        raise HTTPException(status_code=503, detail=status["error"])
    
    # Get individual device states using general function
    pdu_state = grpc_client.get_device_state_general("pdu")
    gpio_state_general = grpc_client.get_device_state_general("gpio")
    detector_state = grpc_client.get_device_state_general("detector")
    motion_state = grpc_client.get_device_state_general("motion")
    
    # Get GPIO button states (key switch and enable button)
    gpio_buttons = grpc_client.get_gpio_button_states()
    
    # Get calibration status (stub for now)
    calibration = CalibrationStatus(
        id=None,
        timestamp=None,
        valid=True,  # TODO: Get from hardware server
        expires_at=None,
        distance_check=True,
        snr_threshold=10.0
    )
    
    # Map hardware interlocks to API model
    interlocks_data = status.get("interlocks", {})
    interlocks = SafetyInterlocks(
        overall_safe=interlocks_data.get("overall_safe", False),
        key_switch=gpio_buttons.get("key_switch", False),  # From GPIO
        enable_button=gpio_buttons.get("enable_button", False),  # From GPIO
        door_closed=interlocks_data.get("door_closed", False),
        emergency_stop=interlocks_data.get("emergency_stop", False),
        radiation_safe=interlocks_data.get("radiation_safe", False),
        cooling_ok=interlocks_data.get("cooling_ok", False),
        power_ok=interlocks_data.get("power_ok", False)
    )
    
    # Get detailed detector info for temperature/voltage
    device_state = grpc_client.get_device_state()
    detector_data = device_state.get("detector", {})
    motion_data = device_state.get("motion", {})
    
    return SystemHealth(
        state=status.get("state", "UNKNOWN"),
        interlocks=interlocks,
        calibration=calibration,
        pdu={
            "powered": pdu_state.get("powered", False),
            "status": pdu_state.get("status", "Off"),
            "outputs": pdu_state.get("outputs", {}),
        } if "error" not in pdu_state else None,
        gpio={
            "powered": gpio_state_general.get("powered", False),
            "status": gpio_state_general.get("status", "Off"),
        } if "error" not in gpio_state_general else None,
        detector={
            "powered": detector_state.get("powered", False),
            "status": detector_state.get("status", "OFF"),
            "temperature": detector_data.get("temperature"),
            "voltage": detector_data.get("voltage"),
        } if "error" not in detector_state else None,
        motion={
            "powered": motion_state.get("powered", False),
            "status": motion_state.get("status", "OFF"),
            "is_homed": motion_data.get("is_homed"),
            # Map structured position to a single float or None per schema
            "position": (motion_data.get("position", {}) or {}).get("x") if isinstance(motion_data.get("position"), dict) else motion_data.get("position"),
        } if "error" not in motion_state else None,
        uptime=0,  # TODO: Calculate from server start time
        cloud_connected=False,  # TODO: Implement cloud status
        last_heartbeat=datetime.utcnow().isoformat()
    )


# ============================================================================
# Patient Endpoints
# ============================================================================

from .models import PatientCreate, PatientResponse

@app.post("/api/patients", response_model=PatientResponse)
async def create_patient(
    patient: PatientCreate,
    user: Dict = Depends(get_current_user)
):
    """
    Create a new patient record in the orchestrator database.
    
    IMPORTANT: Patient PII stays in orchestrator database only.
    """
    try:
        # Generate UUID for patient
        patient_id = str(uuid.uuid4())
        
        # Parse date of birth
        dob = datetime.fromisoformat(patient.date_of_birth)
        
        # Create patient record
        patient_record = PatientRecord(
            patient_id=patient_id,
            first_name=patient.first_name,
            last_name=patient.last_name,
            date_of_birth=dob,
            medical_record_number=patient.medical_record_number,
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow()
        )
        
        db.create_patient(patient_record)
        
        return PatientResponse(
            patient_id=patient_id,
            first_name=patient.first_name,
            last_name=patient.last_name,
            date_of_birth=patient.date_of_birth,
            medical_record_number=patient.medical_record_number,
            created_at=patient_record.created_at.isoformat()
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create patient: {str(e)}")


@app.get("/api/patients/search", response_model=PatientResponse)
async def search_patient(
    mrn: str,
    user: Dict = Depends(get_current_user)
):
    """
    Search for a patient by Medical Record Number.
    
    Returns 404 if patient not found.
    """
    patient = db.find_patient_by_mrn(mrn)
    
    if not patient:
        raise HTTPException(status_code=404, detail=f"Patient not found with MRN: {mrn}")
    
    return PatientResponse(
        patient_id=patient.patient_id,
        first_name=patient.first_name,
        last_name=patient.last_name,
        date_of_birth=patient.date_of_birth.isoformat(),
        medical_record_number=patient.medical_record_number,
        created_at=patient.created_at.isoformat()
    )


@app.get("/api/patients/{patient_id}", response_model=PatientResponse)
async def get_patient_by_id(patient_id: str, user: Dict = Depends(get_current_user)):
    """Load a patient by ID (used by UI after selection)."""
    patient = db.get_patient(patient_id)
    if not patient:
        raise HTTPException(status_code=404, detail=f"Patient not found: {patient_id}")
    return PatientResponse(
        patient_id=patient.patient_id,
        first_name=patient.first_name,
        last_name=patient.last_name,
        date_of_birth=patient.date_of_birth.isoformat(),
        medical_record_number=patient.medical_record_number,
        created_at=patient.created_at.isoformat()
    )


# ============================================================================
# Measurement Endpoints
# ============================================================================

@app.get("/api/measurement_hb")
async def measurement_heartbeat(user: Dict = Depends(get_current_user)):
    """Return current measurement/calibration heartbeats for UI.
    
    Response example:
    {
      "runs": [
        {"run_id": "UUID", "type": "measurement", "elapsed": 3, "total_secs": 60, "percent": 5, "status": "running", "started": "ISO"},
        {"run_id": "UUID", "type": "calibration", "elapsed": 7, "percent": null, "status": "running", "started": "ISO"}
      ],
      "timestamp": "ISO"
    }
    """
    runs = []
    now_iso = datetime.utcnow().isoformat()
    for run_id, data in active_runs.items():
        total = data.get("total_secs")
        elapsed = data.get("elapsed", 0)
        percent = int((elapsed / total) * 100) if total and total > 0 else None
        runs.append({
            "run_id": run_id,
            "type": data.get("type"),
            "elapsed": elapsed,
            "total_secs": total,
            "percent": percent,
            "status": data.get("status", "running"),
            "started": data.get("started"),
        })
    return {"runs": runs, "timestamp": now_iso}

@app.post("/api/measurements/start", response_model=MeasurementStartResponse)
async def start_measurement(
    request: MeasurementStartRequest,
    user: Dict = Depends(get_current_user)
):
    """
    Start a new measurement.
    
    Privacy protection flow:
    1. Get patient data from local database (includes PII)
    2. Generate UUID for measurement (no PII)
    3. Store UUID → patient mapping locally
    4. Send ONLY UUID to hardware server via gRPC
    5. Return measurement ID to Web UI
    """
    # Verify patient exists in local database
    patient = db.get_patient(request.patient_id)
    if not patient:
        return MeasurementStartResponse(
            success=False,
            error=f"Patient not found: {request.patient_id}"
        )
    
    # Generate UUID for measurement (privacy protection)
    measurement_uuid = str(uuid.uuid4())
    
    # Store measurement record locally (links UUID to patient)
    measurement = MeasurementRecord(
        measurement_id=measurement_uuid,
        patient_id=patient.patient_id,
        timestamp=datetime.utcnow(),
        operator_id=user["user_id"],
        measurement_type="patient",
        result_data=None,
        qc_status="pending",
        uploaded_to_cloud=False,
        upload_timestamp=None
    )
    db.record_measurement(measurement)
    
    # Call hardware server with UUID only (NO patient PII)
    result = await grpc_client.start_exposure_with_uuid(
        measurement_id=measurement_uuid,
        operator_id=user["user_id"],
        exposure_time_ms=request.exposure_duration
    )
    
    if "error" in result:
        return MeasurementStartResponse(
            success=False,
            error=result["error"]
        )
    
    # Broadcast measurement start (with countdown info)
    await broadcast_event({
        "type": "measurement_start",
        "data": {
            "measurement_id": measurement_uuid,
            "patient_name": f"{patient.first_name} {patient.last_name}",
            "timestamp": measurement.timestamp.isoformat(),
            "exposure_duration_ms": request.exposure_duration,
            "countdown_start": True
        }
    })

    # Broadcast safety status hint for UI (orange 'Measurement')
    await broadcast_event({
        "type": "safety_status",
        "data": {
            "label": "Measurement",
            "color": "Orange",
            "radiation_safe": False
        }
    })
    
    return MeasurementStartResponse(
        success=True,
        measurement_id=measurement_uuid,
        run_id=measurement_uuid
    )


@app.post("/api/measurements/{run_id}/stop")
async def stop_measurement(run_id: str, user: Dict = Depends(get_current_user)):
    """Stop ongoing measurement (safe; no activation button required)."""
    result = grpc_client.stop_measurement(user=user["user_id"])
    
    if "error" in result:
        raise HTTPException(status_code=500, detail=result["error"])
    
    # Broadcast update to WebSocket clients
    await broadcast_event({
        "type": "measurement_stop",
        "data": {
            "run_id": run_id,
            "status": "stopped",
            "user": user["user_id"]
        }
    })
    
    return {"status": "stopped", "run_id": run_id}


@app.post("/api/measurements/{run_id}/abort")
async def abort_measurement(run_id: str, user: Dict = Depends(get_current_user)):
    """Emergency abort measurement."""
    # TODO: Implement abort in gRPC client
    return {"status": "aborted", "run_id": run_id}


@app.get("/api/measurements", response_model=MeasurementHistoryResponse)
async def get_measurement_history(
    limit: int = 50,
    user: Dict = Depends(get_current_user)
):
    """
    Get measurement history with patient information.
    
    Note: This endpoint joins local patient data with measurement records.
    Only orchestrator has access to patient identifiable information.
    """
    # TODO: Implement query from database
    # For now, return empty list
    return MeasurementHistoryResponse(
        measurements=[],
        total=0
    )


# ============================================================================
# Calibration Endpoints
# ============================================================================

async def run_calibration_async(calibration_id: str, user_id: str):
    """Background task to run calibration and broadcast results via WebSocket.
    
    Includes timeout protection: 1.3x expected integration time.
    Typical calibration: 5-10 seconds, timeout: ~13 seconds max.
    """
    global current_calibration_id
    
    # Timeout: 1.3x typical calibration time (assume ~10s max integration)
    # This prevents hanging forever if hardware server crashes
    CALIBRATION_TIMEOUT = 13.0  # seconds (1.3 * 10s)
    
    try:
        print(f"🔬 Starting calibration {calibration_id} for user {user_id} (timeout: {CALIBRATION_TIMEOUT}s)")
        
        # Broadcast start event
        await broadcast_event({
            "type": "calibration_start",
            "data": {
                "calibration_id": calibration_id,
                "user": user_id,
                "status": "running"
            }
        })
        
        # Run calibration with timeout protection
        try:
            result = await asyncio.wait_for(
                asyncio.to_thread(
                    grpc_client.calibrate_detector,
                    user=user_id
                ),
                timeout=CALIBRATION_TIMEOUT
            )
        except asyncio.TimeoutError:
            error_msg = f"Calibration timed out after {CALIBRATION_TIMEOUT}s - hardware server may be unresponsive"
            print(f"❌ {error_msg}")
            raise Exception(error_msg)
        
        print(f"🔬 Calibration {calibration_id} completed with result: {result.get('success')}")
        
        # Persist to database if successful
        if not result.get("error") and result.get("success"):
            try:
                qc = result.get("qc_checks", {})
                poni = qc.get("poni", {})
                db_cal = DBCalibrationRecord(
                    calibration_id=calibration_id,
                    timestamp=datetime.fromisoformat(result.get("timestamp", datetime.utcnow().isoformat())),
                    operator_id=user_id,
                    calibrant_material=result.get("calibrant_material"),
                    overall_pass=bool(result.get("overall_pass", False)),
                    total_intensity=(qc.get("total_intensity", {}) or {}).get("measured"),
                    total_intensity_threshold=(qc.get("total_intensity", {}) or {}).get("threshold"),
                    total_intensity_pass=(qc.get("total_intensity", {}) or {}).get("passed"),
                    goodness_of_fit=(qc.get("goodness_of_fit", {}) or {}).get("measured"),
                    goodness_of_fit_threshold=(qc.get("goodness_of_fit", {}) or {}).get("threshold"),
                    goodness_of_fit_pass=(qc.get("goodness_of_fit", {}) or {}).get("passed"),
                    snr=(qc.get("snr", {}) or {}).get("measured"),
                    snr_threshold=(qc.get("snr", {}) or {}).get("threshold"),
                    snr_pass=(qc.get("snr", {}) or {}).get("passed"),
                    ring_quality=(qc.get("ring_quality", {}) or {}).get("measured"),
                    ring_quality_threshold=(qc.get("ring_quality", {}) or {}).get("threshold"),
                    ring_quality_pass=(qc.get("ring_quality", {}) or {}).get("passed"),
                    distance_mm=poni.get("distance_mm"),
                    beam_center_x=poni.get("beam_center_x"),
                    beam_center_y=poni.get("beam_center_y"),
                    wavelength_angstrom=poni.get("wavelength_angstrom"),
                    expires_at=None,
                    invalidated_at=None,
                    invalidation_reason=None,
                    formatted_report=result.get("formatted_report")
                )
                db.record_calibration(db_cal)
                print(f"✅ Calibration {calibration_id} persisted to database")
            except Exception as e:
                print(f"❌ Calibration persistence error: {e}")
        
        # Broadcast completion event
        await broadcast_event({
            "type": "calibration_complete",
            "data": {
                "calibration_id": calibration_id,
                "success": result.get("success", False),
                "overall_pass": result.get("overall_pass"),
                "error": result.get("error"),
                "user": user_id
            }
        })
        
    except Exception as e:
        print(f"❌ Calibration {calibration_id} error: {e}")
        # Broadcast error
        await broadcast_event({
            "type": "calibration_complete",
            "data": {
                "calibration_id": calibration_id,
                "success": False,
                "error": str(e),
                "user": user_id
            }
        })
    finally:
        current_calibration_id = None
        print(f"🔬 Calibration task {calibration_id} finished")


@app.post("/api/calibration/start", response_model=CalibrationStartResponse)
async def start_calibration(user: Dict = Depends(get_current_user)):
    """Start daily calibration procedure (async - returns immediately).
    
    Calibration runs in background and notifies via WebSocket:
    - calibration_start: when calibration begins
    - calibration_complete: when finished (success or error)
    
    Timeout protection: 13 seconds (1.3x typical integration time).
    If hardware server is unresponsive, task auto-fails with timeout error.
    """
    global active_calibration_task, current_calibration_id
    
    # Check if calibration already running
    if active_calibration_task and not active_calibration_task.done():
        return CalibrationStartResponse(
            success=False,
            error="Calibration already in progress"
        )
    
    calibration_id = str(uuid.uuid4())
    current_calibration_id = calibration_id
    
    # Start calibration in background
    active_calibration_task = asyncio.create_task(
        run_calibration_async(calibration_id, user["user_id"])
    )
    
    print(f"🔬 Calibration {calibration_id} started in background")
    
    return CalibrationStartResponse(
        success=True,
        calibration_id=calibration_id
    )


@app.get("/api/calibration/status", response_model=CalibrationStatus)
async def get_calibration_status(user: Dict = Depends(get_current_user)):
    """Get current calibration status."""
    # TODO: Get from hardware server
    return CalibrationStatus(
        id=str(uuid.uuid4()),
        timestamp=datetime.utcnow().isoformat(),
        valid=True,
        expires_at=(datetime.utcnow()).isoformat(),
        distance_check=True,
        snr_threshold=10.0
    )


@app.get("/api/calibration/latest")
async def get_latest_calibration(user: Dict = Depends(get_current_user)):
    """Get the latest calibration QC report as JSON.

    Response schema:
    {
      "success": bool,
      "calibration_id": str | null,
      "timestamp": str | null,
      "calibrant_material": str | null,
      "overall_pass": bool | null,
      "formatted_report": str | null,
      "error": str | null
    }
    """
    # Prefer persisted calibration (has calibration_id)
    cal = db.get_current_calibration()
    if cal:
        # Reconstruct QC checks structure from DB fields if available
        qc_checks = {
            "total_intensity": {
                "passed": cal.total_intensity_pass,
                "measured": cal.total_intensity,
                "threshold": cal.total_intensity_threshold,
            },
            "goodness_of_fit": {
                "passed": cal.goodness_of_fit_pass,
                "measured": cal.goodness_of_fit,
                "threshold": cal.goodness_of_fit_threshold,
            },
            "snr": {
                "passed": cal.snr_pass,
                "measured": cal.snr,
                "threshold": cal.snr_threshold,
            },
            "ring_quality": {
                "passed": cal.ring_quality_pass,
                "measured": cal.ring_quality,
                "threshold": cal.ring_quality_threshold,
            },
            "poni": {
                "distance_mm": cal.distance_mm,
                "beam_center_x": cal.beam_center_x,
                "beam_center_y": cal.beam_center_y,
                "wavelength_angstrom": cal.wavelength_angstrom,
            }
        }
        payload = {
            "success": True,
            "calibration_id": cal.calibration_id,
            "timestamp": cal.timestamp.isoformat(),
            "calibrant_material": cal.calibrant_material,
            "overall_pass": cal.overall_pass,
            "qc_checks": qc_checks,
            "formatted_report": cal.formatted_report,
            "error": None,
        }
        return JSONResponse(content=payload)

    # Fallback to hardware server
    result = grpc_client.get_last_calibration()
    if "error" in result:
        return JSONResponse(content={
            "success": False,
            "calibration_id": None,
            "timestamp": None,
            "calibrant_material": None,
            "overall_pass": None,
            "formatted_report": None,
            "error": result.get("error")
        })
    if not result.get("has_calibration"):
        return JSONResponse(content={
            "success": False,
            "calibration_id": None,
            "timestamp": None,
            "calibrant_material": None,
            "overall_pass": None,
            "formatted_report": None,
            "error": "No calibration available"
        })

    payload = {
        "success": True,
        "calibration_id": result.get("calibration_id"),  # May be None if server doesn't provide it
        "timestamp": result.get("timestamp"),
        "calibrant_material": result.get("calibrant_material"),
        "overall_pass": result.get("overall_pass"),
        "qc_checks": result.get("qc_checks"),
        "formatted_report": result.get("formatted_report"),
        "error": None,
    }
    return JSONResponse(content=payload)


@app.get("/api/calibration/running")
async def get_running_calibration(user: Dict = Depends(get_current_user)):
    """Check if calibration is currently running.
    
    Returns:
        - running: bool - Whether calibration is in progress
        - calibration_id: str | None - ID of running calibration
    """
    global active_calibration_task, current_calibration_id
    
    is_running = (
        active_calibration_task is not None 
        and not active_calibration_task.done()
    )
    
    return {
        "running": is_running,
        "calibration_id": current_calibration_id if is_running else None
    }


# ============================================================================
# Calibration History Endpoint
# ============================================================================

@app.get("/api/calibration/history")
async def get_calibration_history(hours: int = 24, limit: int = 50, user: Dict = Depends(get_current_user)):
    """Return calibration records from the last N hours (default 24), newest first.
    Response:
    {
      "success": true,
      "total": <int>,
      "records": [ { "id", "timestamp", "operator", "material", "overall_pass" } ]
    }
    """
    try:
        records = db.list_calibrations(limit=limit, since_hours=hours)
        items = [
            {
                "id": r.calibration_id,
                "timestamp": r.timestamp.isoformat(),
                "operator": r.operator_id,
                "material": r.calibrant_material,
                "overall_pass": r.overall_pass,
            }
            for r in records
        ]
        return JSONResponse(content={
            "success": True,
            "total": len(items),
            "records": items,
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch calibration history: {e}")


# ============================================================================
# GPIO and Safety Endpoints
# ============================================================================

@app.get("/api/gpio/state")
async def get_gpio_state(user: Dict = Depends(get_current_user)):
    """
    Get GPIO state including buttons, interlocks, and LEDs.
    
    Returns:
        - powered: bool
        - key_switch_on: bool
        - activation_button_active: bool
        - activation_remaining_secs: Optional[int]
        - interlocks: dict
        - main_led: str
        - radiation_led: str
    """
    gpio_state = grpc_client.get_gpio_state()
    
    if "error" in gpio_state:
        raise HTTPException(status_code=503, detail=gpio_state["error"])
    
    return gpio_state


@app.get("/api/gpio/enable-button")
async def check_enable_button(user: Dict = Depends(get_current_user)):
    """
    Check enable button status.
    
    Returns:
        - active: bool - Whether button is currently active
        - remaining_secs: Optional[int] - Seconds remaining in 20s window
    """
    button_status = grpc_client.check_enable_button()
    
    if "error" in button_status:
        raise HTTPException(status_code=503, detail=button_status["error"])
    
    return button_status


# ============================================================================
# Hardware Control Endpoints
# ============================================================================

@app.post("/api/hardware/{device}/init")
async def initialize_device(device: str, user: Dict = Depends(get_current_user)):
    """
    Initialize hardware device (detector or motion).
    
    REQUIRES: Enable button must be active (within 20s window).
    
    This is a HARMFUL operation that requires operator confirmation via
    the physical enable button before initialization can proceed.
    
    Returns:
        - 400: Invalid device type
        - 412: Precondition failed (enable button not active)
        - 500: Server error
        - 200: Success
    """
    if device not in ["detector", "motion"]:
        raise HTTPException(
            status_code=400, 
            detail=f"Invalid device type: {device}. Must be 'detector' or 'motion'"
        )
    
    # Check enable button first
    button_check = grpc_client.check_enable_button()
    if not button_check.get("active", False):
        raise HTTPException(
            status_code=412,
            detail={
                "error": "Enable button not active",
                "message": f"Click ENABLE button in GPIO panel to initialize {device}",
                "instructions": "You have 20 seconds after clicking the button to complete initialization"
            }
        )
    
    # Initialize the device
    if device == "detector":
        result = grpc_client.initialize_detector(user=user["user_id"])
    else:  # motion
        result = grpc_client.initialize_motion(user=user["user_id"])
    
    if "error" in result:
        # Check if it's an enable button error
        if "Enable button" in result["error"]:
            raise HTTPException(status_code=412, detail=result)
        else:
            raise HTTPException(status_code=500, detail=result["error"])
    
    # Broadcast update to WebSocket clients
    await broadcast_event({
        "type": "hardware_init",
        "data": {
            "device": device,
            "status": "initialized",
            "user": user["user_id"],
            "detail": result
        }
    })
    
    return {"success": True, "device": device, "status": "initialized", "detail": result}


@app.post("/api/hardware/{device}/stop")
async def stop_device(device: str, user: Dict = Depends(get_current_user)):
    """
    Power off hardware device (detector or motion).
    
    This is a SAFE operation - no enable button required.
    Powers down the device in a controlled manner.
    
    Note: This is different from stopping an active motion. To stop motion movement without
    powering off the motion controller, use POST /api/motion/stop.
    """
    if device not in ["detector", "motion"]:
        raise HTTPException(
            status_code=400, 
            detail=f"Invalid device type: {device}. Must be 'detector' or 'motion'"
        )
    
    # Power off the device (safe operation, no button needed)
    if device == "detector":
        result = grpc_client.power_off_detector(user=user["user_id"])
    else:  # motion
        result = grpc_client.power_off_motion(user=user["user_id"])
    
    if "error" in result:
        raise HTTPException(status_code=500, detail=result["error"])
    
    # Broadcast update to WebSocket clients
    await broadcast_event({
        "type": "hardware_stop",
        "data": {
            "device": device,
            "status": "powered_off",
            "user": user["user_id"]
        }
    })
    
    return {"success": True, "device": device, "status": "powered_off"}


# ============================================================================
# Motion Control Endpoints (Safe operations)
# ============================================================================

@app.post("/api/motion/stop")
async def stop_motion(user: Dict = Depends(get_current_user)):
    """Stop any ongoing motion immediately (safe; no activation button required)."""
    result = grpc_client.stop_motion(user=user["user_id"])
    if "error" in result:
        raise HTTPException(status_code=500, detail=result["error"])

    # Broadcast update to WebSocket clients
    await broadcast_event({
        "type": "motion_stop",
        "data": {
            "status": "stopped",
            "user": user["user_id"]
        }
    })

    return {"status": "stopped"}


# ============================================================================
# Command Discovery Endpoints (No auth required - for diagnostics)
# ============================================================================

@app.get("/api/v1/server/capabilities")
async def get_server_capabilities():
    """Get hardware server capabilities and version."""
    global grpc_client
    return grpc_client.get_server_capabilities()


@app.get("/api/v1/server/commands")
async def list_server_commands(service: Optional[str] = None):
    """List all available commands from hardware server."""
    global grpc_client
    data = grpc_client.list_commands()
    
    if "error" in data:
        raise HTTPException(status_code=503, detail=data["error"])
    
    # Filter by service if requested
    if service:
        commands = data.get("commands", [])
        data["commands"] = [cmd for cmd in commands if cmd["service_name"] == service]
    
    return data


@app.get("/api/v1/server/commands/readiness")
async def list_server_commands_with_readiness(service: Optional[str] = None, user: Dict = Depends(get_current_user)):
    """List commands and their readiness for execution based on current hardware state.

    This endpoint is intended for the UI to enable/disable actions.
    """
    global grpc_client

    # Fetch available commands and current state snapshots
    commands_resp = grpc_client.list_commands()
    if "error" in commands_resp:
        raise HTTPException(status_code=503, detail=commands_resp["error"])

    full_state = grpc_client.get_full_server_state()
    if "error" in full_state:
        raise HTTPException(status_code=503, detail=full_state["error"])

    gpio = grpc_client.get_gpio_state()
    if "error" in gpio:
        # Fallback to partial GPIO from full_state if present
        gpio = full_state.get("gpio", {})

    detector = full_state.get("detector", {})

    # Helper: normalize current safety values
    safety_state = (full_state.get("safety_state") or "UNKNOWN").upper()
    key_on = bool(gpio.get("key_switch_on") or gpio.get("interlocks", {}).get("key_switch"))
    enable_active = bool(gpio.get("activation_button_active") or gpio.get("interlocks", {}).get("enable_button"))
    interlocks = gpio.get("interlocks", {}) or {}

    def detector_ready() -> bool:
        status = (detector.get("status") or "").upper()
        # Treat IDLE as ready; INIT/EXPOSING/READING/ERROR are not ready
        return bool(detector.get("powered")) and status == "DETECTOR_IDLE" or status == "IDLE"

    def check_requirements(cmd: Dict) -> Dict:
        reasons = []
        svc = cmd.get("service_name")
        name = cmd.get("command_name")
        reqs = set((cmd.get("safety_requirements") or []))

        # Generic requirements based on advertised safety_requirements
        if "key_switch_on" in reqs and not key_on:
            reasons.append("Key switch is OFF")
        if "activation_button" in reqs and not enable_active:
            reasons.append("Enable button is not active")
        if "interlocks_safe" in reqs and not interlocks.get("overall_safe", False):
            reasons.append("Interlocks are not satisfied")
        if "beam_closed" in reqs and not interlocks.get("radiation_safe", False):
            reasons.append("Beam is open (radiation not safe)")
        if "state_armed_or_running" in reqs and safety_state not in ("PENDING_ARMED", "RUNNING"):
            reasons.append(f"State is {safety_state}, requires PENDING_ARMED or RUNNING")

        # Command-specific refinements derived from server schemas/rules
        if (svc, name) == ("Acquisition", "CalibrateDetector"):
            # Allowed states: IDLE or LOCKED per schema
            if safety_state not in ("IDLE", "LOCKED"):
                reasons.append(f"State {safety_state} not allowed for calibration")
            # Interlock details per schema
            if not interlocks.get("door_closed", False):
                reasons.append("Safety door must be closed")
            if not interlocks.get("emergency_stop", False):
                reasons.append("Emergency stop is pressed")
            # Detector readiness
            if not detector_ready():
                reasons.append("Detector is not ready")
        elif (svc, name) == ("DeviceInitialization", "InitializeDetector"):
            if not interlocks.get("radiation_safe", False):
                reasons.append("Beam must be closed (radiation safe=true)")
            if not interlocks.get("cooling_ok", False):
                reasons.append("Cooling not OK")
            if not interlocks.get("power_ok", False):
                reasons.append("Power not OK")
        elif (svc, name) == ("DeviceInitialization", "InitializeMotion"):
            if not interlocks.get("radiation_safe", False):
                reasons.append("Beam must be closed (radiation safe=true)")
            if not interlocks.get("power_ok", False):
                reasons.append("Power not OK")

        ready = len(reasons) == 0
        return {
            "service_name": svc,
            "command_name": name,
            "description": cmd.get("description"),
            "ready": ready,
            "reasons": reasons,
        }

    commands = commands_resp.get("commands", [])
    if service:
        commands = [c for c in commands if c.get("service_name") == service]

    readiness_list = [check_requirements(cmd) for cmd in commands]
    return {"commands": readiness_list, "timestamp": datetime.utcnow().isoformat()} 


@app.post("/api/v1/server/validate-compatibility")
async def validate_compatibility(
    client_version: str = "0.1.0",
    protocol_version: str = "1.0.0",
):
    """Validate compatibility between orchestrator and hardware server."""
    global grpc_client
    result = grpc_client.validate_compatibility(
        client_version=client_version,
        protocol_version=protocol_version,
    )
    
    if "error" in result:
        raise HTTPException(status_code=503, detail=result["error"])
    
    return result


# ============================================================================
# WebSocket for Real-time Updates
# ============================================================================

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """
    WebSocket connection for real-time system updates.
    
    Clients receive:
    - System state changes
    - Measurement progress updates
    - Safety alerts
    - Calibration status changes
    """
    await websocket.accept()
    websocket_connections.append(websocket)
    
    try:
        # Send initial connection confirmation
        await websocket.send_json({
            "type": "connection",
            "data": {"status": "connected"},
            "timestamp": datetime.utcnow().isoformat()
        })
        
        # Keep connection alive and listen for client messages
        while True:
            data = await websocket.receive_text()
            # Echo back for now (can implement client commands later)
            await websocket.send_json({
                "type": "echo",
                "data": {"message": data},
                "timestamp": datetime.utcnow().isoformat()
            })
            
    except WebSocketDisconnect:
        websocket_connections.remove(websocket)
    except Exception as e:
        print(f"WebSocket error: {e}")
        if websocket in websocket_connections:
            websocket_connections.remove(websocket)


async def handle_run_notification(change_type: str):
    """Handle RUN notification (START/TICK/DONE) from hardware server."""
    global active_runs
    
    try:
        parts = change_type.split(":")
        if len(parts) < 3:
            return
            
        event_type = parts[0]  # START, TICK, DONE
        run_type = parts[1]    # measurement, calibration
        run_id = parts[2]      # UUID
        
        if event_type == "START":
            # START:measurement:uuid:total_secs OR START:calibration:uuid
            total_secs = int(parts[3]) if len(parts) > 3 and parts[3].isdigit() else None
            active_runs[run_id] = {
                "type": run_type,
                "started": datetime.utcnow().isoformat(),
                "total_secs": total_secs,
                "elapsed": 0,
                "status": "running"
            }
            print(f"📊 Started tracking {run_type} {run_id}")
            
        elif event_type == "TICK":
            if run_id in active_runs:
                if run_type == "measurement" and len(parts) > 3:
                    # TICK:measurement:uuid:elapsed/total
                    elapsed_total = parts[3].split("/")
                    if len(elapsed_total) == 2:
                        active_runs[run_id]["elapsed"] = int(elapsed_total[0])
                elif run_type == "calibration":
                    # TICK:calibration:uuid (just increment)
                    active_runs[run_id]["elapsed"] = active_runs[run_id].get("elapsed", 0) + 1
                    
        elif event_type == "DONE":
            if run_id in active_runs:
                status = parts[3] if len(parts) > 3 else "completed"
                active_runs[run_id]["status"] = status
                print(f"📊 Completed tracking {run_type} {run_id}: {status}")
                
                # For measurements, try to get result
                if run_type == "measurement":
                    # Broadcast measurement completion
                    await broadcast_event({
                        "type": "measurement_complete",
                        "data": {
                            "run_id": run_id,
                            "status": status
                        }
                    })
                    
                # For calibrations, broadcast calibration result
                elif run_type == "calibration":
                    await broadcast_event({
                        "type": "calibration_complete",
                        "data": {
                            "run_id": run_id,
                            "status": status,
                            "overall_pass": status == "passed"
                        }
                    })
                    
    except Exception as e:
        print(f"❌ Error handling RUN notification {change_type}: {e}")
        
        
async def broadcast_event(event: Dict):
    """Broadcast event to all connected WebSocket clients."""
    event["timestamp"] = datetime.utcnow().isoformat()
    
    disconnected = []
    for ws in websocket_connections:
        try:
            await ws.send_json(event)
        except Exception:
            disconnected.append(ws)
    
    # Remove disconnected clients
    for ws in disconnected:
        websocket_connections.remove(ws)


# ============================================================================
# Error Handlers
# ============================================================================

@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Custom HTTP exception handler."""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": str(exc.detail),
            "code": str(exc.status_code)
        }
    )


# ============================================================================
# Development Helper Endpoints
# ============================================================================

@app.get("/api/connection/status")
async def connection_status():
    """Check connection status to hardware server (no auth required)."""
    # Try to get system status from hardware server
    if grpc_client is None:
        return {
            "connected": False,
            "error": "gRPC client not initialized"
        }
    
    try:
        # Quick health check to hardware server
        status = grpc_client.get_status()
        if "error" in status:
            return {
                "connected": False,
                "error": status.get("error", "Unknown error")
            }
        
        return {
            "connected": True,
            "hardware_state": status.get("state", "UNKNOWN"),
            "hardware_ok": status.get("ok", False)
        }
    except Exception as e:
        return {
            "connected": False,
            "error": str(e)
        }


@app.get("/api/debug/status")
async def debug_status():
    """Debug endpoint to check server status (no auth required)."""
    return {
        "status": "running",
        "grpc_connected": grpc_client is not None,
        "db_initialized": db is not None,
        "auth_manager": auth_manager is not None,
        "active_sessions": len(auth_manager.active_sessions) if auth_manager else 0,
        "websocket_clients": len(websocket_connections)
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "omniscan_orchestrator.rest_server:app",
        host="0.0.0.0",
        port=8080,
        reload=True,
        log_level="info"
    )
