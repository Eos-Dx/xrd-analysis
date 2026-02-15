"""
Pytest Configuration and Shared Fixtures
Medical Device Software - FDA Compliant Testing
"""

import pytest
import tempfile
import shutil
from pathlib import Path
from datetime import datetime, timezone
from unittest.mock import Mock, AsyncMock
import uuid

from omniscan_orchestrator.database import (
    OrchestratorDatabase,
    PatientRecord,
    MeasurementRecord,
    UserSession
)
from omniscan_orchestrator.audit import AuditLogger, initialize_audit_logger
from omniscan_orchestrator.auth_manager import AuthenticationManager
from omniscan_orchestrator.rbac import User, Role


# ==============================================================================
# Temporary Directory Fixtures
# ==============================================================================

@pytest.fixture
def temp_dir():
    """Create temporary directory for test isolation."""
    temp_path = tempfile.mkdtemp()
    yield Path(temp_path)
    shutil.rmtree(temp_path, ignore_errors=True)


@pytest.fixture
def temp_db_path(temp_dir):
    """Temporary database path."""
    return str(temp_dir / "test_orchestrator.db")


@pytest.fixture
def temp_audit_dir(temp_dir):
    """Temporary audit log directory."""
    audit_dir = temp_dir / "audit_logs"
    audit_dir.mkdir(exist_ok=True)
    return audit_dir


# ==============================================================================
# Database Fixtures
# ==============================================================================

@pytest.fixture
def db(temp_db_path):
    """Initialize clean test database."""
    database = OrchestratorDatabase(temp_db_path)
    yield database
    # Cleanup handled by temp_dir


@pytest.fixture
def sample_patient():
    """Sample patient record for testing."""
    return PatientRecord(
        patient_id=str(uuid.uuid4()),
        first_name="John",
        last_name="Doe",
        date_of_birth=datetime(1980, 1, 1),
        medical_record_number="MRN123456",
        created_at=datetime.utcnow(),
        updated_at=datetime.utcnow()
    )


@pytest.fixture
def sample_measurement():
    """Sample measurement record for testing."""
    return MeasurementRecord(
        measurement_id=str(uuid.uuid4()),
        patient_id=str(uuid.uuid4()),
        timestamp=datetime.utcnow(),
        operator_id="OP001",
        measurement_type="patient",
        result_data=None,
        qc_status="pending",
        uploaded_to_cloud=False,
        upload_timestamp=None
    )


# ==============================================================================
# Audit Logger Fixtures
# ==============================================================================

@pytest.fixture
def audit_logger(temp_audit_dir):
    """Initialize test audit logger."""
    device_id = "TEST_DEVICE_001"
    logger = AuditLogger(temp_audit_dir, device_id)
    yield logger


# ==============================================================================
# gRPC Client Mock Fixtures
# ==============================================================================

@pytest.fixture
def mock_grpc_client():
    """Mock gRPC client for testing without hardware."""
    client = Mock()
    
    # Mock key switch state (ON by default for testing)
    key_switch_response = Mock()
    key_switch_response.key_on = True
    client.get_key_switch_state = AsyncMock(return_value=key_switch_response)
    
    # Mock device state
    state_response = Mock()
    state_response.state = "IDLE"
    state_response.calibration_valid = True
    state_response.calibration_timestamp = datetime.utcnow().isoformat()
    client.get_device_state = AsyncMock(return_value=state_response)
    
    # Mock calibration status
    cal_response = Mock()
    cal_response.is_valid = True
    cal_response.last_calibration = datetime.utcnow().isoformat()
    cal_response.hours_until_expiry = 20.0
    client.get_calibration_status = AsyncMock(return_value=cal_response)
    
    # Mock measurement start
    measure_response = Mock()
    measure_response.success = True
    measure_response.measurement_id = str(uuid.uuid4())
    client.start_exposure_with_uuid = AsyncMock(return_value=measure_response)
    
    return client


@pytest.fixture
def mock_grpc_client_key_off(mock_grpc_client):
    """Mock gRPC client with key switch OFF."""
    key_switch_response = Mock()
    key_switch_response.key_on = False
    mock_grpc_client.get_key_switch_state = AsyncMock(return_value=key_switch_response)
    return mock_grpc_client


@pytest.fixture
def mock_grpc_client_calibration_expired(mock_grpc_client):
    """Mock gRPC client with expired calibration."""
    cal_response = Mock()
    cal_response.is_valid = False
    cal_response.last_calibration = datetime(2025, 10, 20).isoformat()
    cal_response.hours_until_expiry = -5.0
    mock_grpc_client.get_calibration_status = AsyncMock(return_value=cal_response)
    return mock_grpc_client


# ==============================================================================
# Authentication Manager Fixtures
# ==============================================================================

@pytest.fixture
def auth_manager(db, mock_grpc_client):
    """Authentication manager with mocked gRPC."""
    return AuthenticationManager(mock_grpc_client, db)


# ==============================================================================
# RBAC User Fixtures
# ==============================================================================

@pytest.fixture
def clinical_operator():
    """Clinical operator user."""
    return User(
        user_id="OP001",
        username="operator:OP001",
        role=Role.CLINICAL_OPERATOR,
        certificate_cn="operator:OP001",
        authenticated_at=datetime.now(timezone.utc)
    )


@pytest.fixture
def maintenance_engineer():
    """Maintenance engineer user."""
    return User(
        user_id="ENG001",
        username="engineer:ENG001",
        role=Role.MAINTENANCE_ENGINEER,
        certificate_cn="engineer:ENG001",
        authenticated_at=datetime.now(timezone.utc)
    )


@pytest.fixture
def administrator():
    """Administrator user."""
    return User(
        user_id="ADMIN001",
        username="admin:ADMIN001",
        role=Role.ADMINISTRATOR,
        certificate_cn="admin:ADMIN001",
        authenticated_at=datetime.now(timezone.utc)
    )


# ==============================================================================
# Test Data Helpers
# ==============================================================================

def create_test_patients(db, count=5):
    """Helper to create multiple test patients."""
    patients = []
    for i in range(count):
        patient = PatientRecord(
            patient_id=str(uuid.uuid4()),
            first_name=f"Patient{i}",
            last_name=f"Test{i}",
            date_of_birth=datetime(1980 + i, 1, 1),
            medical_record_number=f"MRN{100000 + i}",
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow()
        )
        db.create_patient(patient)
        patients.append(patient)
    return patients


def create_test_measurements(db, patient_id, count=3):
    """Helper to create multiple test measurements."""
    measurements = []
    for i in range(count):
        measurement = MeasurementRecord(
            measurement_id=str(uuid.uuid4()),
            patient_id=patient_id,
            timestamp=datetime.utcnow(),
            operator_id=f"OP{i:03d}",
            measurement_type="patient",
            result_data=None,
            qc_status="pass" if i % 2 == 0 else "pending",
            uploaded_to_cloud=False,
            upload_timestamp=None
        )
        db.record_measurement(measurement)
        measurements.append(measurement)
    return measurements


# ==============================================================================
# Pytest Hooks
# ==============================================================================

def pytest_configure(config):
    """Configure pytest with custom markers and settings."""
    config.addinivalue_line(
        "markers", "fda_critical: Critical tests for FDA compliance"
    )
