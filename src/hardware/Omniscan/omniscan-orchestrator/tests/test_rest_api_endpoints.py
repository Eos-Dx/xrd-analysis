"""
Comprehensive Test Suite for Orchestrator REST API Endpoints
Tests all REST endpoints with proper mocking and validation.
"""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from datetime import datetime, timedelta
import uuid
import json

from omniscan_orchestrator.rest_server import app
from omniscan_orchestrator.database import (
    OrchestratorDatabase,
    PatientRecord,
    MeasurementRecord,
    CalibrationRecord,
    UserSession
)


# ==============================================================================
# Fixtures
# ==============================================================================

@pytest.fixture
def mock_grpc_client():
    """Mock gRPC client for all endpoints."""
    client = Mock()
    
    # Mock get_status
    client.get_status.return_value = {
        "state": "IDLE",
        "ok": True,
        "interlocks": {
            "overall_safe": True,
            "door_closed": True,
            "emergency_stop": False,
            "radiation_safe": True,
            "cooling_ok": True,
            "power_ok": True
        }
    }
    
    # Mock get_full_server_state
    client.get_full_server_state.return_value = {
        "safety_state": "IDLE",
        "detector": {
            "powered": True,
            "initialized": True,
            "status": "IDLE",
            "temperature": -20.5,
            "total_exposures": 42
        },
        "motion": {
            "powered": True,
            "initialized": True,
            "status": "IDLE",
            "is_homed": True,
            "position_x": 100.0,
            "position_y": 200.0,
            "total_moves": 15
        }
    }
    
    # Mock get_gpio_state
    client.get_gpio_state.return_value = {
        "powered": True,
        "key_switch_on": True,
        "activation_button_active": False,
        "activation_remaining_secs": 0,
        "interlocks": {
            "overall_safe": True,
            "door_closed": True,
            "emergency_stop": False,
            "radiation_safe": True,
            "cooling_ok": True,
            "power_ok": True
        },
        "main_led": "GREEN",
        "radiation_led": "OFF"
    }
    
    # Mock get_device_state_general
    def mock_device_state(device):
        return {
            "powered": True,
            "status": "IDLE" if device in ["detector", "motion"] else "On"
        }
    client.get_device_state_general.side_effect = mock_device_state
    
    # Mock get_device_state
    client.get_device_state.return_value = {
        "detector": {"temperature": -20.5, "voltage": 5.0},
        "motion": {"is_homed": True, "position": {"x": 100.0, "y": 200.0}}
    }
    
    # Mock get_gpio_button_states
    client.get_gpio_button_states.return_value = {
        "key_switch": True,
        "enable_button": False
    }
    
    # Mock check_enable_button
    client.check_enable_button.return_value = {
        "active": False,
        "remaining_secs": 0
    }
    
    # Mock start_exposure_with_uuid (async)
    async def mock_start_exposure(*args, **kwargs):
        return {
            "success": True,
            "measurement_id": str(uuid.uuid4())
        }
    client.start_exposure_with_uuid = mock_start_exposure
    
    # Mock stop_measurement
    client.stop_measurement.return_value = {"success": True}
    
    # Mock calibrate_detector
    client.calibrate_detector.return_value = {
        "success": True,
        "calibration_id": str(uuid.uuid4()),
        "timestamp": datetime.utcnow().isoformat(),
        "calibrant_material": "LaB6",
        "overall_pass": True,
        "qc_checks": {
            "total_intensity": {"measured": 1500, "threshold": 1000, "passed": True},
            "goodness_of_fit": {"measured": 0.95, "threshold": 0.90, "passed": True},
            "snr": {"measured": 15.0, "threshold": 10.0, "passed": True},
            "ring_quality": {"measured": 0.98, "threshold": 0.85, "passed": True},
            "poni": {
                "distance_mm": 150.0,
                "beam_center_x": 512.0,
                "beam_center_y": 512.0,
                "wavelength_angstrom": 1.54
            }
        },
        "formatted_report": "Calibration Report\n==================\n\nAll checks passed."
    }
    
    # Mock get_last_calibration
    client.get_last_calibration.return_value = {
        "has_calibration": True,
        "calibration_id": str(uuid.uuid4()),
        "timestamp": datetime.utcnow().isoformat(),
        "calibrant_material": "LaB6",
        "overall_pass": True,
        "formatted_report": "Calibration Report\n==================\n\nAll checks passed."
    }
    
    # Mock initialize_detector
    client.initialize_detector.return_value = {"success": True, "message": "Detector initialized"}
    
    # Mock initialize_motion
    client.initialize_motion.return_value = {"success": True, "message": "Motion initialized"}
    
    # Mock power_off_detector
    client.power_off_detector.return_value = {"success": True, "message": "Detector powered off"}
    
    # Mock power_off_motion
    client.power_off_motion.return_value = {"success": True, "message": "Motion powered off"}
    
    # Mock stop_motion
    client.stop_motion.return_value = {"success": True, "message": "Motion stopped"}
    
    # Mock CommandDiscovery methods
    client.get_server_capabilities.return_value = {
        "server_version": "0.1.0",
        "protocol_version": "1.0.0",
        "build_info": "Test Build",
        "supported_features": ["command_discovery", "state_streaming"]
    }
    
    client.list_commands.return_value = {
        "commands": [
            {
                "service_name": "omniscan.OmniscanService",
                "command_name": "GetStatus",
                "description": "Get system status"
            },
            {
                "service_name": "omniscan.OmniscanService",
                "command_name": "InitializeDetector",
                "description": "Initialize detector"
            }
        ]
    }
    
    client.validate_compatibility.return_value = {
        "compatible": True,
        "message": "Client and server are compatible",
        "missing_commands": [],
        "version_warnings": []
    }
    
    # Mock subscribe_to_state_updates (returns None for testing)
    client.subscribe_to_state_updates.return_value = None
    
    # Mock close
    client.close.return_value = None
    
    return client


@pytest.fixture
def mock_db(tmp_path):
    """Mock database for testing."""
    db_path = tmp_path / "test_orchestrator.db"
    db = OrchestratorDatabase(str(db_path))
    return db


@pytest.fixture
def mock_auth_manager(mock_grpc_client, mock_db):
    """Mock authentication manager."""
    from omniscan_orchestrator.auth_manager import AuthenticationManager
    auth_mgr = AuthenticationManager(mock_grpc_client, mock_db)
    
    # Create a test session
    session_id = "test-session-123"
    session = UserSession(
        session_id=session_id,
        user_id="test_operator",
        role="operator",
        login_time=datetime.utcnow(),
        logout_time=None,
        last_activity=datetime.utcnow()
    )
    auth_mgr.active_sessions[session_id] = session
    mock_db.record_login("test_operator", session_id, "operator")
    
    return auth_mgr


@pytest.fixture
def client(mock_grpc_client, mock_db, mock_auth_manager):
    """FastAPI test client with all dependencies mocked."""
    with patch('omniscan_orchestrator.rest_server.grpc_client', mock_grpc_client), \
         patch('omniscan_orchestrator.rest_server.db', mock_db), \
         patch('omniscan_orchestrator.rest_server.auth_manager', mock_auth_manager), \
         patch('omniscan_orchestrator.rest_server.state_monitor_task', None):
        
        # Use TestClient without lifespan to avoid startup issues
        with TestClient(app) as test_client:
            yield test_client


@pytest.fixture
def auth_headers():
    """Authentication headers for authenticated requests."""
    return {"X-Session-Id": "test-session-123"}


@pytest.fixture
def sample_patient(mock_db):
    """Create a sample patient in the database."""
    patient = PatientRecord(
        patient_id=str(uuid.uuid4()),
        first_name="John",
        last_name="Doe",
        date_of_birth=datetime(1980, 1, 1),
        medical_record_number="MRN123456",
        created_at=datetime.utcnow(),
        updated_at=datetime.utcnow()
    )
    mock_db.create_patient(patient)
    return patient


# ==============================================================================
# Authentication Endpoints Tests
# ==============================================================================

class TestAuthenticationEndpoints:
    """Test authentication endpoints."""
    
    def test_login_success(self, client, mock_grpc_client):
        """Test successful login."""
        response = client.post(
            "/api/auth/login",
            json={"username": "operator1", "password": "password123"}
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "session_id" in data
        assert data["user_id"] == "operator1"
        assert data["role"] == "operator"
    
    def test_login_key_switch_off(self, client, mock_grpc_client):
        """Test login fails when key switch is off."""
        # Mock key switch off
        mock_grpc_client.get_gpio_state.return_value = {
            "key_switch_on": False,
            "interlocks": {"overall_safe": False}
        }
        
        response = client.post(
            "/api/auth/login",
            json={"username": "operator1", "password": "password123"}
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is False
        assert "error" in data
    
    def test_logout_success(self, client, auth_headers):
        """Test successful logout."""
        response = client.post("/api/auth/logout", headers=auth_headers)
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "Logged out successfully" in data["message"]
    
    def test_logout_no_session(self, client):
        """Test logout without session returns 401."""
        response = client.post("/api/auth/logout")
        
        assert response.status_code == 401


# ==============================================================================
# System Health Endpoints Tests
# ==============================================================================

class TestSystemHealthEndpoints:
    """Test system health and state endpoints."""
    
    def test_get_system_state(self, client, auth_headers):
        """Test GET /api/state."""
        response = client.get("/api/state", headers=auth_headers)
        
        assert response.status_code == 200
        data = response.json()
        assert "state" in data
        assert "interlocks" in data
        assert "devices" in data
        assert "calibration" in data
        assert data["state"] == "IDLE"
        assert data["interlocks"]["overall_safe"] is True
        assert "detector" in data["devices"]
        assert "motion" in data["devices"]
    
    def test_get_system_health(self, client, auth_headers):
        """Test GET /api/health."""
        response = client.get("/api/health", headers=auth_headers)
        
        assert response.status_code == 200
        data = response.json()
        assert "state" in data
        assert "interlocks" in data
        assert "calibration" in data
        assert "pdu" in data
        assert "gpio" in data
        assert "detector" in data
        assert "motion" in data
        assert data["state"] == "IDLE"
    
    def test_get_system_state_no_auth(self, client):
        """Test GET /api/state without authentication."""
        response = client.get("/api/state")
        
        assert response.status_code == 401


# ==============================================================================
# Patient Endpoints Tests
# ==============================================================================

class TestPatientEndpoints:
    """Test patient management endpoints."""
    
    def test_create_patient(self, client, auth_headers):
        """Test POST /api/patients."""
        patient_data = {
            "first_name": "Jane",
            "last_name": "Smith",
            "date_of_birth": "1990-05-15",
            "medical_record_number": "MRN789012"
        }
        
        response = client.post(
            "/api/patients",
            json=patient_data,
            headers=auth_headers
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "patient_id" in data
        assert data["first_name"] == "Jane"
        assert data["last_name"] == "Smith"
        assert data["medical_record_number"] == "MRN789012"
    
    def test_create_patient_invalid_date(self, client, auth_headers):
        """Test POST /api/patients with invalid date format."""
        patient_data = {
            "first_name": "Jane",
            "last_name": "Smith",
            "date_of_birth": "invalid-date",
            "medical_record_number": "MRN789012"
        }
        
        response = client.post(
            "/api/patients",
            json=patient_data,
            headers=auth_headers
        )
        
        assert response.status_code == 400
    
    def test_search_patient_by_mrn(self, client, auth_headers, sample_patient):
        """Test GET /api/patients/search."""
        response = client.get(
            f"/api/patients/search?mrn={sample_patient.medical_record_number}",
            headers=auth_headers
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["patient_id"] == sample_patient.patient_id
        assert data["first_name"] == sample_patient.first_name
        assert data["last_name"] == sample_patient.last_name
    
    def test_search_patient_not_found(self, client, auth_headers):
        """Test GET /api/patients/search with non-existent MRN."""
        response = client.get(
            "/api/patients/search?mrn=NONEXISTENT",
            headers=auth_headers
        )
        
        assert response.status_code == 404
    
    def test_create_patient_no_auth(self, client):
        """Test POST /api/patients without authentication."""
        patient_data = {
            "first_name": "Jane",
            "last_name": "Smith",
            "date_of_birth": "1990-05-15",
            "medical_record_number": "MRN789012"
        }
        
        response = client.post("/api/patients", json=patient_data)
        
        assert response.status_code == 401


# ==============================================================================
# Measurement Endpoints Tests
# ==============================================================================

class TestMeasurementEndpoints:
    """Test measurement endpoints."""
    
    def test_start_measurement(self, client, auth_headers, sample_patient, mock_grpc_client):
        """Test POST /api/measurements/start."""
        measurement_data = {
            "patient_id": sample_patient.patient_id,
            "exposure_duration": 5000
        }
        
        response = client.post(
            "/api/measurements/start",
            json=measurement_data,
            headers=auth_headers
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "measurement_id" in data
        assert "run_id" in data
    
    def test_start_measurement_patient_not_found(self, client, auth_headers):
        """Test POST /api/measurements/start with non-existent patient."""
        measurement_data = {
            "patient_id": str(uuid.uuid4()),
            "exposure_duration": 5000
        }
        
        response = client.post(
            "/api/measurements/start",
            json=measurement_data,
            headers=auth_headers
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is False
        assert "error" in data
    
    def test_stop_measurement(self, client, auth_headers):
        """Test POST /api/measurements/{run_id}/stop."""
        run_id = str(uuid.uuid4())
        
        response = client.post(
            f"/api/measurements/{run_id}/stop",
            headers=auth_headers
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "stopped"
        assert data["run_id"] == run_id
    
    def test_abort_measurement(self, client, auth_headers):
        """Test POST /api/measurements/{run_id}/abort."""
        run_id = str(uuid.uuid4())
        
        response = client.post(
            f"/api/measurements/{run_id}/abort",
            headers=auth_headers
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "aborted"
    
    def test_get_measurement_history(self, client, auth_headers):
        """Test GET /api/measurements."""
        response = client.get("/api/measurements", headers=auth_headers)
        
        assert response.status_code == 200
        data = response.json()
        assert "measurements" in data
        assert "total" in data
        assert isinstance(data["measurements"], list)


# ==============================================================================
# Calibration Endpoints Tests
# ==============================================================================

class TestCalibrationEndpoints:
    """Test calibration endpoints."""
    
    def test_start_calibration(self, client, auth_headers, mock_grpc_client):
        """Test POST /api/calibration/start."""
        response = client.post("/api/calibration/start", headers=auth_headers)
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "calibration_id" in data
    
    def test_start_calibration_failure(self, client, auth_headers, mock_grpc_client):
        """Test POST /api/calibration/start with gRPC failure."""
        mock_grpc_client.calibrate_detector.return_value = {
            "error": "Calibration failed",
            "success": False
        }
        
        response = client.post("/api/calibration/start", headers=auth_headers)
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is False
        assert "error" in data
    
    def test_get_calibration_status(self, client, auth_headers):
        """Test GET /api/calibration/status."""
        response = client.get("/api/calibration/status", headers=auth_headers)
        
        assert response.status_code == 200
        data = response.json()
        assert "id" in data
        assert "timestamp" in data
        assert "valid" in data
        assert data["valid"] is True
    
    def test_get_latest_calibration(self, client, auth_headers, mock_grpc_client):
        """Test GET /api/calibration/latest."""
        response = client.get("/api/calibration/latest", headers=auth_headers)
        
        assert response.status_code == 200
        data = response.json()
        assert "calibration_id" in data
        assert "timestamp" in data
        assert "overall_pass" in data
        assert "formatted_report" in data
    
    def test_get_latest_calibration_not_found(self, client, auth_headers, mock_grpc_client):
        """Test GET /api/calibration/latest when no calibration exists."""
        mock_grpc_client.get_last_calibration.return_value = {
            "has_calibration": False
        }
        
        response = client.get("/api/calibration/latest", headers=auth_headers)
        
        assert response.status_code == 404


# ==============================================================================
# GPIO and Safety Endpoints Tests
# ==============================================================================

class TestGPIOEndpoints:
    """Test GPIO and safety endpoints."""
    
    def test_get_gpio_state(self, client, auth_headers):
        """Test GET /api/gpio/state."""
        response = client.get("/api/gpio/state", headers=auth_headers)
        
        assert response.status_code == 200
        data = response.json()
        assert "powered" in data
        assert "key_switch_on" in data
        assert "activation_button_active" in data
        assert "interlocks" in data
        assert "main_led" in data
        assert "radiation_led" in data
    
    def test_check_enable_button(self, client, auth_headers):
        """Test GET /api/gpio/enable-button."""
        response = client.get("/api/gpio/enable-button", headers=auth_headers)
        
        assert response.status_code == 200
        data = response.json()
        assert "active" in data
        assert "remaining_secs" in data
    
    def test_get_gpio_state_no_auth(self, client):
        """Test GET /api/gpio/state without authentication."""
        response = client.get("/api/gpio/state")
        
        assert response.status_code == 401


# ==============================================================================
# Hardware Control Endpoints Tests
# ==============================================================================

class TestHardwareControlEndpoints:
    """Test hardware control endpoints."""
    
    def test_initialize_detector_button_inactive(self, client, auth_headers, mock_grpc_client):
        """Test POST /api/hardware/detector/init without enable button."""
        response = client.post(
            "/api/hardware/detector/init",
            headers=auth_headers
        )
        
        assert response.status_code == 412  # Precondition failed
    
    def test_initialize_detector_button_active(self, client, auth_headers, mock_grpc_client):
        """Test POST /api/hardware/detector/init with enable button active."""
        # Mock enable button active
        mock_grpc_client.check_enable_button.return_value = {
            "active": True,
            "remaining_secs": 15
        }
        
        response = client.post(
            "/api/hardware/detector/init",
            headers=auth_headers
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["device"] == "detector"
        assert data["status"] == "initialized"
    
    def test_initialize_motion(self, client, auth_headers, mock_grpc_client):
        """Test POST /api/hardware/motion/init."""
        # Mock enable button active
        mock_grpc_client.check_enable_button.return_value = {
            "active": True,
            "remaining_secs": 15
        }
        
        response = client.post(
            "/api/hardware/motion/init",
            headers=auth_headers
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["device"] == "motion"
    
    def test_initialize_invalid_device(self, client, auth_headers):
        """Test POST /api/hardware/{device}/init with invalid device."""
        response = client.post(
            "/api/hardware/invalid_device/init",
            headers=auth_headers
        )
        
        assert response.status_code == 400
    
    def test_stop_detector(self, client, auth_headers):
        """Test POST /api/hardware/detector/stop."""
        response = client.post(
            "/api/hardware/detector/stop",
            headers=auth_headers
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["device"] == "detector"
        assert data["status"] == "powered_off"
    
    def test_stop_motion(self, client, auth_headers):
        """Test POST /api/hardware/motion/stop."""
        response = client.post(
            "/api/hardware/motion/stop",
            headers=auth_headers
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["device"] == "motion"


# ==============================================================================
# Motion Control Endpoints Tests
# ==============================================================================

class TestMotionControlEndpoints:
    """Test motion control endpoints."""
    
    def test_stop_motion_movement(self, client, auth_headers):
        """Test POST /api/motion/stop."""
        response = client.post("/api/motion/stop", headers=auth_headers)
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "stopped"


# ==============================================================================
# Command Discovery Endpoints Tests
# ==============================================================================

class TestCommandDiscoveryEndpoints:
    """Test command discovery endpoints (no auth required)."""
    
    def test_get_server_capabilities(self, client):
        """Test GET /api/v1/server/capabilities."""
        response = client.get("/api/v1/server/capabilities")
        
        assert response.status_code == 200
        data = response.json()
        assert "server_version" in data
        assert "protocol_version" in data
        assert "build_info" in data
        assert "supported_features" in data
    
    def test_list_server_commands(self, client):
        """Test GET /api/v1/server/commands."""
        response = client.get("/api/v1/server/commands")
        
        assert response.status_code == 200
        data = response.json()
        assert "commands" in data
        assert isinstance(data["commands"], list)
        assert len(data["commands"]) > 0
    
    def test_list_server_commands_filtered(self, client):
        """Test GET /api/v1/server/commands with service filter."""
        response = client.get(
            "/api/v1/server/commands?service=omniscan.OmniscanService"
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "commands" in data
        # All commands should be from the filtered service
        for cmd in data["commands"]:
            assert cmd["service_name"] == "omniscan.OmniscanService"
    
    def test_validate_compatibility(self, client):
        """Test POST /api/v1/server/validate-compatibility."""
        response = client.post(
            "/api/v1/server/validate-compatibility",
            params={
                "client_version": "0.1.0",
                "protocol_version": "1.0.0"
            }
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "compatible" in data
        assert data["compatible"] is True
        assert "message" in data


# ==============================================================================
# WebSocket Tests
# ==============================================================================

class TestWebSocketEndpoint:
    """Test WebSocket endpoint."""
    
    def test_websocket_connection(self, client):
        """Test WebSocket connection at /ws."""
        with client.websocket_connect("/ws") as websocket:
            # Receive connection confirmation
            data = websocket.receive_json()
            assert data["type"] == "connection"
            assert data["data"]["status"] == "connected"
            
            # Send a test message
            websocket.send_text("Hello")
            
            # Receive echo response
            echo = websocket.receive_json()
            assert echo["type"] == "echo"
            assert echo["data"]["message"] == "Hello"


# ==============================================================================
# Development Helper Endpoints Tests
# ==============================================================================

class TestDevelopmentEndpoints:
    """Test development helper endpoints."""
    
    def test_connection_status(self, client):
        """Test GET /api/connection/status (no auth required)."""
        response = client.get("/api/connection/status")
        
        assert response.status_code == 200
        data = response.json()
        assert "connected" in data
        assert data["connected"] is True
        assert "hardware_state" in data
    
    def test_debug_status(self, client):
        """Test GET /api/debug/status (no auth required)."""
        response = client.get("/api/debug/status")
        
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert data["status"] == "running"
        assert "grpc_connected" in data
        assert "db_initialized" in data
        assert "auth_manager" in data


# ==============================================================================
# Error Handling Tests
# ==============================================================================

class TestErrorHandling:
    """Test error handling and edge cases."""
    
    def test_grpc_error_handling(self, client, auth_headers, mock_grpc_client):
        """Test handling of gRPC errors."""
        mock_grpc_client.get_full_server_state.return_value = {
            "error": "Connection lost"
        }
        
        response = client.get("/api/state", headers=auth_headers)
        
        assert response.status_code == 503
    
    def test_invalid_session_header(self, client):
        """Test with invalid session ID."""
        response = client.get(
            "/api/state",
            headers={"X-Session-Id": "invalid-session-id"}
        )
        
        assert response.status_code == 401
    
    def test_missing_required_fields(self, client, auth_headers):
        """Test POST with missing required fields."""
        response = client.post(
            "/api/patients",
            json={"first_name": "Jane"},  # Missing required fields
            headers=auth_headers
        )
        
        assert response.status_code == 422  # Validation error


# ==============================================================================
# Integration Tests
# ==============================================================================

class TestIntegrationScenarios:
    """Test complete workflow scenarios."""
    
    def test_complete_measurement_workflow(self, client, auth_headers, sample_patient, mock_grpc_client):
        """Test complete measurement workflow from start to stop."""
        # Mock enable button active for initialization
        mock_grpc_client.check_enable_button.return_value = {
            "active": True,
            "remaining_secs": 15
        }
        
        # 1. Check system state
        response = client.get("/api/state", headers=auth_headers)
        assert response.status_code == 200
        
        # 2. Initialize detector
        response = client.post(
            "/api/hardware/detector/init",
            headers=auth_headers
        )
        assert response.status_code == 200
        
        # 3. Start measurement
        response = client.post(
            "/api/measurements/start",
            json={
                "patient_id": sample_patient.patient_id,
                "exposure_duration": 5000
            },
            headers=auth_headers
        )
        assert response.status_code == 200
        data = response.json()
        run_id = data["run_id"]
        
        # 4. Stop measurement
        response = client.post(
            f"/api/measurements/{run_id}/stop",
            headers=auth_headers
        )
        assert response.status_code == 200
        
        # 5. Stop detector
        response = client.post(
            "/api/hardware/detector/stop",
            headers=auth_headers
        )
        assert response.status_code == 200
    
    def test_calibration_workflow(self, client, auth_headers, mock_grpc_client):
        """Test complete calibration workflow."""
        # 1. Check current calibration status
        response = client.get("/api/calibration/status", headers=auth_headers)
        assert response.status_code == 200
        
        # 2. Start calibration
        response = client.post("/api/calibration/start", headers=auth_headers)
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        
        # 3. Get latest calibration
        response = client.get("/api/calibration/latest", headers=auth_headers)
        assert response.status_code == 200
        data = response.json()
        assert data["overall_pass"] is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
