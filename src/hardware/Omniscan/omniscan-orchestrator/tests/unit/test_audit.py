"""
Unit Tests for Audit Trail Module
Tests: UR-AUDIT-001 (Comprehensive Audit Trail), SYS_OMNI-SERVER_007 (Audit Database)
Per VER_OMNI-SERVER_004, VER_OMNI-SERVER_005: Data retention and audit validation
"""

import pytest
from pathlib import Path
import json
from datetime import datetime, timezone
import hashlib

from omniscan_orchestrator.audit import AuditLogger, AuditEvent


@pytest.mark.unit
@pytest.mark.audit
class TestAuditEventCreation:
    """Test audit event creation per UR-AUDIT-001."""
    
    def test_audit_event_creation(self):
        """Test audit event is created with all required fields."""
        event = AuditEvent(
            event_type="measurement",
            user_id="OP001",
            description="Measurement executed",
            device_id="DEV001",
            details={"measurement_id": "123"},
            session_id="sess_001"
        )
        
        assert event.event_id is not None
        assert event.timestamp is not None
        assert event.event_type == "measurement"
        assert event.user_id == "OP001"
        assert event.device_id == "DEV001"
        assert event.session_id == "sess_001"
    
    def test_audit_event_to_dict(self):
        """Test audit event serialization."""
        event = AuditEvent(
            event_type="authentication",
            user_id="OP001",
            description="Login successful"
        )
        
        event_dict = event.to_dict()
        
        assert isinstance(event_dict, dict)
        assert "event_id" in event_dict
        assert "timestamp" in event_dict
        assert "event_type" in event_dict
        assert "user_id" in event_dict
    
    def test_audit_event_hash_computation(self):
        """Test cryptographic hash computation per UR-AUDIT-001."""
        event = AuditEvent(
            event_type="measurement",
            user_id="OP001",
            description="Test event"
        )
        
        hash1 = event.compute_hash()
        
        assert hash1 is not None
        assert len(hash1) == 64  # SHA-256 produces 64-character hex string
        
        # Hash should be deterministic
        hash2 = event.compute_hash()
        assert hash1 == hash2


@pytest.mark.unit
@pytest.mark.audit
class TestAuditLoggerBasics:
    """Test audit logger basic functionality."""
    
    def test_logger_initialization(self, temp_audit_dir):
        """Test audit logger initializes correctly."""
        logger = AuditLogger(temp_audit_dir, "TEST_DEV_001")
        
        assert logger.log_directory == temp_audit_dir
        assert logger.device_id == "TEST_DEV_001"
        assert logger.current_log_file.exists()
    
    def test_log_file_creation(self, temp_audit_dir):
        """Test log file is created with correct naming."""
        logger = AuditLogger(temp_audit_dir, "TEST_DEV_001")
        
        date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        expected_filename = f"audit_{date_str}.jsonl"
        
        assert logger.current_log_file.name == expected_filename
    
    def test_log_event_basic(self, audit_logger):
        """Test basic event logging."""
        event_id = audit_logger.log(
            event_type="measurement",
            user_id="OP001",
            description="Test measurement"
        )
        
        assert event_id is not None
        
        # Verify file was written
        assert audit_logger.current_log_file.exists()
        
        # Read and verify content
        with open(audit_logger.current_log_file, 'r') as f:
            line = f.readline()
            entry = json.loads(line)
            
            assert entry["event_id"] == event_id
            assert entry["event_type"] == "measurement"
            assert entry["user_id"] == "OP001"


@pytest.mark.unit
@pytest.mark.audit
class TestAuditHashChaining:
    """Test cryptographic hash chaining per UR-AUDIT-001."""
    
    def test_hash_chain_integrity(self, audit_logger):
        """Test that events form a cryptographic chain."""
        # Log first event
        audit_logger.log("measurement", "OP001", "Event 1")
        
        # Log second event
        audit_logger.log("measurement", "OP001", "Event 2")
        
        # Read log file and verify chain
        with open(audit_logger.current_log_file, 'r') as f:
            lines = f.readlines()
            
            event1 = json.loads(lines[0])
            event2 = json.loads(lines[1])
            
            # First event should have no previous hash
            assert event1["previous_hash"] is None
            
            # Second event should link to first
            assert event2["previous_hash"] is not None
    
    def test_hash_chain_verification(self, audit_logger):
        """Test audit trail integrity verification."""
        # Log multiple events
        for i in range(5):
            audit_logger.log("measurement", f"OP{i:03d}", f"Event {i}")
        
        # Verify integrity
        is_valid, error = audit_logger.verify_integrity()
        
        assert is_valid is True
        assert error is None
    
    def test_hash_chain_tampering_detection(self, audit_logger, temp_audit_dir):
        """Test that tampering is detected."""
        # Log some events
        audit_logger.log("measurement", "OP001", "Event 1")
        audit_logger.log("measurement", "OP001", "Event 2")
        audit_logger.log("measurement", "OP001", "Event 3")
        
        # Tamper with the middle event
        with open(audit_logger.current_log_file, 'r') as f:
            lines = f.readlines()
        
        # Modify event 2
        event2 = json.loads(lines[1])
        event2["description"] = "TAMPERED"
        lines[1] = json.dumps(event2) + '\n'
        
        # Write back tampered data
        with open(audit_logger.current_log_file, 'w') as f:
            f.writelines(lines)
        
        # Create new logger instance to re-verify
        logger2 = AuditLogger(temp_audit_dir, "TEST_DEV_001")
        is_valid, error = logger2.verify_integrity()
        
        assert is_valid is False
        assert "Hash chain broken" in error


@pytest.mark.unit
@pytest.mark.audit
class TestAuditSpecializedLogging:
    """Test specialized audit logging methods per UR-AUDIT-001."""
    
    def test_log_authentication(self, audit_logger):
        """Test authentication event logging."""
        audit_logger.log_authentication(
            user_id="OP001",
            success=True,
            method="mTLS",
            details={"certificate_cn": "operator:OP001"}
        )
        
        # Verify log entry
        with open(audit_logger.current_log_file, 'r') as f:
            entry = json.loads(f.readline())
            
            assert entry["event_type"] == "authentication"
            assert entry["details"]["success"] is True
            assert entry["details"]["method"] == "mTLS"
    
    def test_log_access_denied(self, audit_logger):
        """Test access denial logging."""
        audit_logger.log_access_denied(
            user_id="OP001",
            resource="/config/modify",
            reason="Insufficient permissions"
        )
        
        with open(audit_logger.current_log_file, 'r') as f:
            entry = json.loads(f.readline())
            
            assert entry["event_type"] == "access_denied"
            assert "Access denied" in entry["description"]
    
    def test_log_config_change(self, audit_logger):
        """Test configuration change logging with before/after values."""
        audit_logger.log_config_change(
            user_id="ENG001",
            key="exposure_time",
            old_value=60,
            new_value=90,
            session_id="sess_001"
        )
        
        with open(audit_logger.current_log_file, 'r') as f:
            entry = json.loads(f.readline())
            
            assert entry["event_type"] == "config_change"
            assert entry["details"]["key"] == "exposure_time"
            assert entry["details"]["old_value"] == 60
            assert entry["details"]["new_value"] == 90
    
    def test_log_maintenance_mode(self, audit_logger):
        """Test maintenance mode tracking per UR-SAFE-002."""
        audit_logger.log_maintenance_mode(
            user_id="ENG001",
            action="entered",
            session_id="sess_maint_001",
            details={"ttl": 900, "bypassed_interlocks": ["door_sensor"]}
        )
        
        with open(audit_logger.current_log_file, 'r') as f:
            entry = json.loads(f.readline())
            
            assert entry["event_type"] == "maintenance_mode"
            assert "entered" in entry["description"]
            assert "bypassed_interlocks" in entry["details"]
    
    def test_log_measurement(self, audit_logger):
        """Test measurement operation logging per UR-AUDIT-001."""
        measurement_params = {
            "exposure_time": 60,
            "beam_intensity": 50,
            "sample_type": "patient"
        }
        
        audit_logger.log_measurement(
            user_id="OP001",
            measurement_id="meas_12345",
            params=measurement_params,
            session_id="sess_001"
        )
        
        with open(audit_logger.current_log_file, 'r') as f:
            entry = json.loads(f.readline())
            
            assert entry["event_type"] == "measurement"
            assert "meas_12345" in entry["description"]
            assert entry["details"]["parameters"] == measurement_params
    
    def test_log_calibration(self, audit_logger):
        """Test calibration event logging per UR-CAL-001."""
        cal_details = {
            "calibrant": "LaB6",
            "peaks_found": 12,
            "quality_score": 0.95
        }
        
        audit_logger.log_calibration(
            user_id="OP001",
            result="passed",
            details=cal_details,
            session_id="sess_001"
        )
        
        with open(audit_logger.current_log_file, 'r') as f:
            entry = json.loads(f.readline())
            
            assert entry["event_type"] == "calibration"
            assert "passed" in entry["description"]
    
    def test_log_safety_status(self, audit_logger):
        """Test safety system status logging per UR-SAFE-001."""
        audit_logger.log_safety_status(
            user_id="SYSTEM",
            status_change="emergency_stop_activated",
            details={
                "trigger": "e_stop_button",
                "beam_state": "off",
                "timestamp": datetime.utcnow().isoformat()
            }
        )
        
        with open(audit_logger.current_log_file, 'r') as f:
            entry = json.loads(f.readline())
            
            assert entry["event_type"] == "safety_status"
            assert entry["user_id"] == "SYSTEM"


@pytest.mark.unit
@pytest.mark.audit
class TestAuditExportAndRetention:
    """Test audit trail export and retention per UR-AUDIT-001."""
    
    def test_export_audit_trail_json(self, audit_logger, temp_audit_dir):
        """Test exporting audit trail to JSON."""
        # Log some events
        for i in range(3):
            audit_logger.log("measurement", f"OP{i:03d}", f"Event {i}")
        
        # Export
        output_path = temp_audit_dir / "export.json"
        date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        
        audit_logger.export_audit_trail(
            start_date=date_str,
            end_date=date_str,
            output_path=output_path,
            format="json"
        )
        
        assert output_path.exists()
        
        # Verify content
        with open(output_path, 'r') as f:
            data = json.load(f)
            
            assert isinstance(data, list)
            assert len(data) == 3
    
    def test_export_audit_trail_csv(self, audit_logger, temp_audit_dir):
        """Test exporting audit trail to CSV."""
        # Log some events
        audit_logger.log("measurement", "OP001", "Event 1")
        audit_logger.log("authentication", "OP002", "Event 2")
        
        # Export
        output_path = temp_audit_dir / "export.csv"
        date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        
        audit_logger.export_audit_trail(
            start_date=date_str,
            end_date=date_str,
            output_path=output_path,
            format="csv"
        )
        
        assert output_path.exists()
        
        # Verify it's valid CSV
        import csv
        with open(output_path, 'r') as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            
            assert len(rows) == 2
            assert "event_type" in rows[0]


@pytest.mark.unit
@pytest.mark.audit
class TestAuditThreadSafety:
    """Test audit logger thread safety."""
    
    def test_concurrent_logging(self, audit_logger):
        """Test that concurrent logging maintains integrity."""
        import threading
        
        def log_events(thread_id):
            for i in range(10):
                audit_logger.log(
                    "measurement",
                    f"THREAD_{thread_id}",
                    f"Event from thread {thread_id}"
                )
        
        threads = []
        for i in range(5):
            t = threading.Thread(target=log_events, args=(i,))
            threads.append(t)
            t.start()
        
        for t in threads:
            t.join()
        
        # Verify all events were logged
        with open(audit_logger.current_log_file, 'r') as f:
            lines = f.readlines()
            
            assert len(lines) == 50  # 5 threads × 10 events
        
        # Verify integrity is maintained
        is_valid, error = audit_logger.verify_integrity()
        assert is_valid is True


@pytest.mark.unit
@pytest.mark.audit
class TestAuditAppendOnly:
    """Test append-only nature of audit logs per UR-AUDIT-001."""
    
    def test_audit_file_append_only(self, audit_logger):
        """Test that audit logger only appends, never modifies existing entries."""
        # Log initial event
        audit_logger.log("measurement", "OP001", "Event 1")
        
        # Read first event
        with open(audit_logger.current_log_file, 'r') as f:
            line1 = f.readline()
            event1 = json.loads(line1)
        
        # Log another event
        audit_logger.log("measurement", "OP001", "Event 2")
        
        # Verify first event unchanged
        with open(audit_logger.current_log_file, 'r') as f:
            line1_after = f.readline()
            
            assert line1 == line1_after
    
    def test_no_modification_methods(self, audit_logger):
        """Verify audit logger has no methods to modify past entries."""
        # Audit logger should not have update or delete methods
        assert not hasattr(audit_logger, 'update_event')
        assert not hasattr(audit_logger, 'delete_event')
        assert not hasattr(audit_logger, 'modify_event')
