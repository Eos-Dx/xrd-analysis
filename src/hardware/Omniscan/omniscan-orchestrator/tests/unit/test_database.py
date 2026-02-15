"""
Unit Tests for Database Module
Tests: UR-DATA-001 (Data Integrity), UR-DATA-002 (Security), UR-DATA-003 (Traceability)
Per VER_OMNI-SERVER_004: Calibration procedure test
"""

import pytest
from datetime import datetime, timedelta
import uuid
import sqlite3

from omniscan_orchestrator.database import (
    OrchestratorDatabase,
    PatientRecord,
    MeasurementRecord,
    UserSession
)


@pytest.mark.unit
@pytest.mark.data_integrity
class TestPatientRecordCRUD:
    """Test patient record CRUD operations per UR-DATA-001."""
    
    def test_create_patient_success(self, db, sample_patient):
        """Test successful patient creation."""
        patient_id = db.create_patient(sample_patient)
        
        assert patient_id == sample_patient.patient_id
        
        # Verify retrieval
        retrieved = db.get_patient(patient_id)
        assert retrieved is not None
        assert retrieved.first_name == sample_patient.first_name
        assert retrieved.last_name == sample_patient.last_name
        assert retrieved.medical_record_number == sample_patient.medical_record_number
    
    def test_create_patient_duplicate_mrn_fails(self, db, sample_patient):
        """Test that duplicate MRN is rejected per UR-DATA-001."""
        db.create_patient(sample_patient)
        
        # Try to create another patient with same MRN
        duplicate = PatientRecord(
            patient_id=str(uuid.uuid4()),
            first_name="Jane",
            last_name="Smith",
            date_of_birth=datetime(1990, 1, 1),
            medical_record_number=sample_patient.medical_record_number,  # Same MRN
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow()
        )
        
        with pytest.raises(ValueError, match="Patient record creation failed"):
            db.create_patient(duplicate)
    
    def test_get_patient_not_found(self, db):
        """Test retrieval of non-existent patient."""
        result = db.get_patient("non-existent-id")
        assert result is None
    
    def test_find_patient_by_mrn(self, db, sample_patient):
        """Test finding patient by medical record number."""
        db.create_patient(sample_patient)
        
        found = db.find_patient_by_mrn(sample_patient.medical_record_number)
        assert found is not None
        assert found.patient_id == sample_patient.patient_id
        assert found.first_name == sample_patient.first_name
    
    def test_find_patient_by_mrn_not_found(self, db):
        """Test MRN lookup for non-existent patient."""
        result = db.find_patient_by_mrn("INVALID_MRN")
        assert result is None
    
    def test_patient_data_persistence(self, db, sample_patient):
        """Test patient data persists across database connections per UR-DATA-001."""
        db.create_patient(sample_patient)
        
        # Create new database connection
        db2 = OrchestratorDatabase(db.db_path)
        retrieved = db2.get_patient(sample_patient.patient_id)
        
        assert retrieved is not None
        assert retrieved.first_name == sample_patient.first_name


@pytest.mark.unit
@pytest.mark.data_integrity
class TestMeasurementRecordOperations:
    """Test measurement record operations per UR-DATA-001, UR-DATA-003."""
    
    def test_record_measurement_success(self, db, sample_patient, sample_measurement):
        """Test successful measurement recording."""
        db.create_patient(sample_patient)
        sample_measurement.patient_id = sample_patient.patient_id
        
        db.record_measurement(sample_measurement)
        
        # Verify retrieval
        retrieved = db.get_measurement(sample_measurement.measurement_id)
        assert retrieved is not None
        assert retrieved.measurement_id == sample_measurement.measurement_id
        assert retrieved.patient_id == sample_patient.patient_id
        assert retrieved.operator_id == sample_measurement.operator_id
    
    def test_measurement_uuid_uniqueness(self, db, sample_patient):
        """Test that measurement IDs are unique per UR-DATA-003."""
        db.create_patient(sample_patient)
        
        measurement_id = str(uuid.uuid4())
        
        m1 = MeasurementRecord(
            measurement_id=measurement_id,
            patient_id=sample_patient.patient_id,
            timestamp=datetime.utcnow(),
            operator_id="OP001",
            measurement_type="patient",
            result_data=None,
            qc_status="pending",
            uploaded_to_cloud=False,
            upload_timestamp=None
        )
        
        db.record_measurement(m1)
        
        # Try to record another measurement with same ID
        m2 = MeasurementRecord(
            measurement_id=measurement_id,  # Same ID
            patient_id=sample_patient.patient_id,
            timestamp=datetime.utcnow(),
            operator_id="OP002",
            measurement_type="patient",
            result_data=None,
            qc_status="pending",
            uploaded_to_cloud=False,
            upload_timestamp=None
        )
        
        with pytest.raises(sqlite3.IntegrityError):
            db.record_measurement(m2)
    
    def test_get_patient_measurements(self, db, sample_patient):
        """Test retrieving all measurements for a patient."""
        db.create_patient(sample_patient)
        
        # Create multiple measurements
        for i in range(3):
            measurement = MeasurementRecord(
                measurement_id=str(uuid.uuid4()),
                patient_id=sample_patient.patient_id,
                timestamp=datetime.utcnow() + timedelta(minutes=i),
                operator_id=f"OP{i:03d}",
                measurement_type="patient",
                result_data=None,
                qc_status="pass" if i % 2 == 0 else "fail",
                uploaded_to_cloud=False,
                upload_timestamp=None
            )
            db.record_measurement(measurement)
        
        measurements = db.get_patient_measurements(sample_patient.patient_id)
        
        assert len(measurements) == 3
        # Should be ordered by timestamp DESC
        assert measurements[0].timestamp >= measurements[1].timestamp
    
    def test_mark_uploaded_to_cloud(self, db, sample_patient, sample_measurement):
        """Test marking measurement as uploaded."""
        db.create_patient(sample_patient)
        sample_measurement.patient_id = sample_patient.patient_id
        db.record_measurement(sample_measurement)
        
        upload_time = datetime.utcnow()
        db.mark_uploaded(sample_measurement.measurement_id, upload_time)
        
        retrieved = db.get_measurement(sample_measurement.measurement_id)
        assert retrieved.uploaded_to_cloud is True
        assert retrieved.upload_timestamp is not None
    
    def test_measurement_data_immutability(self, db, sample_patient, sample_measurement):
        """Test that measurement data is write-once per UR-DATA-001."""
        db.create_patient(sample_patient)
        sample_measurement.patient_id = sample_patient.patient_id
        db.record_measurement(sample_measurement)
        
        # Database should not provide update method for measurements
        # Only upload status can be updated, not measurement data itself
        assert not hasattr(db, 'update_measurement')


@pytest.mark.unit
@pytest.mark.data_integrity
class TestUserSessionManagement:
    """Test user session tracking per UR-USER-001."""
    
    def test_record_login(self, db):
        """Test recording user login."""
        session_id = str(uuid.uuid4())
        user_id = "OP001"
        role = "clinical_operator"
        
        db.record_login(user_id, session_id, role)
        
        session = db.get_session(session_id)
        assert session is not None
        assert session.user_id == user_id
        assert session.role == role
        assert session.logout_time is None
    
    def test_record_logout(self, db):
        """Test recording user logout."""
        session_id = str(uuid.uuid4())
        db.record_login("OP001", session_id, "clinical_operator")
        
        db.record_logout(session_id)
        
        session = db.get_session(session_id)
        assert session.logout_time is not None
    
    def test_update_session_activity(self, db):
        """Test updating session activity timestamp."""
        session_id = str(uuid.uuid4())
        db.record_login("OP001", session_id, "clinical_operator")
        
        session_before = db.get_session(session_id)
        
        # Wait briefly and update
        import time
        time.sleep(0.1)
        db.update_session_activity(session_id)
        
        session_after = db.get_session(session_id)
        assert session_after.last_activity > session_before.last_activity
    
    def test_get_session_not_found(self, db):
        """Test retrieval of non-existent session."""
        result = db.get_session("non-existent-session")
        assert result is None


@pytest.mark.unit
@pytest.mark.data_integrity
class TestDatabaseIntegrity:
    """Test database integrity and foreign key constraints per UR-DATA-001."""
    
    def test_foreign_key_constraint(self, db, sample_measurement):
        """Test that measurements require valid patient_id."""
        # Try to record measurement without creating patient first
        # SQLite with foreign keys enabled should enforce this
        with pytest.raises((sqlite3.IntegrityError, Exception)):
            db.record_measurement(sample_measurement)
    
    def test_database_schema_creation(self, db):
        """Test that all required tables are created."""
        conn = sqlite3.connect(db.db_path)
        cursor = conn.cursor()
        
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [row[0] for row in cursor.fetchall()]
        
        assert "patients" in tables
        assert "measurements" in tables
        assert "user_sessions" in tables
        
        conn.close()
    
    def test_database_indexes_created(self, db):
        """Test that performance indexes are created."""
        conn = sqlite3.connect(db.db_path)
        cursor = conn.cursor()
        
        cursor.execute("SELECT name FROM sqlite_master WHERE type='index'")
        indexes = [row[0] for row in cursor.fetchall()]
        
        # Check for measurement indexes
        index_names = [idx for idx in indexes if 'idx_measurements' in idx]
        assert len(index_names) >= 2  # patient and timestamp indexes
        
        conn.close()
    
    def test_database_file_persistence(self, temp_db_path):
        """Test database file is created and persists per UR-DATA-001."""
        from pathlib import Path
        
        db = OrchestratorDatabase(temp_db_path)
        
        assert Path(temp_db_path).exists()
        assert Path(temp_db_path).is_file()


@pytest.mark.unit
@pytest.mark.data_integrity
class TestDataTraceability:
    """Test complete traceability per UR-DATA-003."""
    
    def test_measurement_complete_metadata(self, db, sample_patient, sample_measurement):
        """Test that measurements capture complete metadata."""
        db.create_patient(sample_patient)
        sample_measurement.patient_id = sample_patient.patient_id
        db.record_measurement(sample_measurement)
        
        retrieved = db.get_measurement(sample_measurement.measurement_id)
        
        # Verify all traceability fields are present
        assert retrieved.measurement_id is not None
        assert retrieved.patient_id is not None
        assert retrieved.timestamp is not None
        assert retrieved.operator_id is not None
        assert retrieved.measurement_type is not None
        assert retrieved.qc_status is not None
    
    def test_patient_linkage_traceability(self, db, sample_patient):
        """Test complete patient-measurement linkage."""
        db.create_patient(sample_patient)
        
        # Create measurement
        measurement = MeasurementRecord(
            measurement_id=str(uuid.uuid4()),
            patient_id=sample_patient.patient_id,
            timestamp=datetime.utcnow(),
            operator_id="OP001",
            measurement_type="patient",
            result_data=None,
            qc_status="pass",
            uploaded_to_cloud=False,
            upload_timestamp=None
        )
        db.record_measurement(measurement)
        
        # Retrieve and verify linkage
        retrieved_measurement = db.get_measurement(measurement.measurement_id)
        retrieved_patient = db.get_patient(retrieved_measurement.patient_id)
        
        assert retrieved_patient.patient_id == sample_patient.patient_id
        assert retrieved_patient.medical_record_number == sample_patient.medical_record_number
