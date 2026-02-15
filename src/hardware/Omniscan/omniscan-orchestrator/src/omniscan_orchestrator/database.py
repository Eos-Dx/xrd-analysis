"""
Database Management for Omniscan Orchestrator

Handles patient records, measurement data, and user sessions.
CRITICAL: Patient PII stays in orchestrator database only - server receives UUID only.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import datetime
from typing import Optional, List
from pathlib import Path
import uuid


@dataclass
class PatientRecord:
    """Patient record with identifiable information (PII).
    
    IMPORTANT: This data NEVER leaves the orchestrator.
    Server only receives UUIDs for measurements.
    """
    patient_id: str
    first_name: str
    last_name: str
    date_of_birth: datetime
    medical_record_number: str
    created_at: datetime
    updated_at: datetime


@dataclass
class MeasurementRecord:
    """Measurement record linking UUID to patient.
    
    The measurement_id UUID is shared with the server.
    All other fields remain in orchestrator only.
    """
    measurement_id: str  # UUID - links to server measurement
    patient_id: str
    timestamp: datetime
    operator_id: str
    measurement_type: str  # "calibration" or "patient"
    sample_id: Optional[str] = None  # Sample identifier
    
    # Calibration context
    calibration_id: Optional[str] = None
    calibration_timestamp: Optional[datetime] = None
    calibration_valid: Optional[bool] = None
    
    # Hardware data
    exposure_duration_ms: Optional[int] = None
    beam_intensity_mean: Optional[float] = None
    beam_intensity_std: Optional[float] = None
    snr: Optional[float] = None
    detector_temperature: Optional[float] = None
    detector_voltage: Optional[float] = None
    
    # Results and QC
    result_data: Optional[bytes] = None  # Processed results
    qc_status: str = "pending"  # "pass", "fail", "pending"
    qc_notes: Optional[str] = None
    clinical_notes: Optional[str] = None
    
    # Upload tracking
    uploaded_to_cloud: bool = False
    upload_timestamp: Optional[datetime] = None


@dataclass
class CalibrationRecord:
    """Calibration record with QC data."""
    calibration_id: str  # UUID from hardware server
    timestamp: datetime
    operator_id: str
    
    # Calibration material and results
    calibrant_material: Optional[str]
    overall_pass: bool
    
    # QC checks
    total_intensity: Optional[float]
    total_intensity_threshold: Optional[float]
    total_intensity_pass: Optional[bool]
    
    goodness_of_fit: Optional[float]
    goodness_of_fit_threshold: Optional[float]
    goodness_of_fit_pass: Optional[bool]
    
    snr: Optional[float]
    snr_threshold: Optional[float]
    snr_pass: Optional[bool]
    
    ring_quality: Optional[float]
    ring_quality_threshold: Optional[float]
    ring_quality_pass: Optional[bool]
    
    # PONI results
    distance_mm: Optional[float]
    beam_center_x: Optional[float]
    beam_center_y: Optional[float]
    wavelength_angstrom: Optional[float]
    
    # Validity
    expires_at: Optional[datetime]
    invalidated_at: Optional[datetime]
    invalidation_reason: Optional[str]
    
    # Full report
    formatted_report: Optional[str]


@dataclass
class UICommandLog:
    """UI command log entry for audit trail."""
    log_id: Optional[int]  # Auto-increment
    timestamp: datetime
    session_id: str
    operator_id: str
    command_type: str
    command_payload: Optional[str]  # JSON
    resource_id: Optional[str]
    result: str  # "success" or "failure"
    error_message: Optional[str]
    ip_address: Optional[str]
    user_agent: Optional[str]


@dataclass
class SystemEventLog:
    """System event log entry."""
    event_id: Optional[int]  # Auto-increment
    timestamp: datetime
    event_type: str
    severity: str  # "INFO", "WARNING", "ERROR", "CRITICAL"
    component: Optional[str]
    message: str
    details: Optional[str]  # JSON
    operator_id: Optional[str]


@dataclass
class UserSession:
    """User session tracking."""
    session_id: str
    user_id: str
    role: str
    login_time: datetime
    logout_time: Optional[datetime]
    last_activity: datetime
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None


class OrchestratorDatabase:
    """SQLite database manager for orchestrator.
    
    Stores patient metadata, measurement records with UUID linkage,
    and user session information. Provides data privacy by keeping
    all patient identifiable information local.
    """
    
    def __init__(self, db_path: str):
        self.db_path = db_path
        self._ensure_directory()
        self._init_database()
    
    def _ensure_directory(self):
        """Ensure database directory exists."""
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
    
    def _init_database(self):
        """Initialize database schema with all required tables."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Schema version tracking
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS schema_version (
                version INTEGER PRIMARY KEY,
                applied_at TEXT NOT NULL,
                description TEXT
            )
        """)
        
        # Patients table - stores all PII
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS patients (
                patient_id TEXT PRIMARY KEY,
                first_name TEXT NOT NULL,
                last_name TEXT NOT NULL,
                date_of_birth TEXT NOT NULL,
                medical_record_number TEXT UNIQUE NOT NULL,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                updated_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Measurements table - links UUIDs to patients with full context
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS measurements (
                measurement_id TEXT PRIMARY KEY,
                patient_id TEXT,
                timestamp TEXT NOT NULL,
                operator_id TEXT NOT NULL,
                measurement_type TEXT NOT NULL,
                sample_id TEXT,
                
                calibration_id TEXT,
                calibration_timestamp TEXT,
                calibration_valid INTEGER,
                
                exposure_duration_ms INTEGER,
                beam_intensity_mean REAL,
                beam_intensity_std REAL,
                snr REAL,
                detector_temperature REAL,
                detector_voltage REAL,
                
                result_data BLOB,
                qc_status TEXT DEFAULT 'pending',
                qc_notes TEXT,
                clinical_notes TEXT,
                
                uploaded_to_cloud INTEGER DEFAULT 0,
                upload_timestamp TEXT,
                
                FOREIGN KEY (patient_id) REFERENCES patients(patient_id),
                FOREIGN KEY (calibration_id) REFERENCES calibration_log(calibration_id)
            )
        """)
        
        # Lightweight migration for existing installations: ensure new columns exist on measurements
        try:
            cursor.execute("PRAGMA table_info(measurements)")
            existing_cols = [row[1] for row in cursor.fetchall()]
            expected_cols = {
                # Optional/context columns only (safe to add with NULL/defaults)
                "sample_id": "TEXT",
                "calibration_id": "TEXT",
                "calibration_timestamp": "TEXT",
                "calibration_valid": "INTEGER",
                "exposure_duration_ms": "INTEGER",
                "beam_intensity_mean": "REAL",
                "beam_intensity_std": "REAL",
                "snr": "REAL",
                "detector_temperature": "REAL",
                "detector_voltage": "REAL",
                "result_data": "BLOB",
                "qc_status": "TEXT DEFAULT 'pending'",
                "qc_notes": "TEXT",
                "clinical_notes": "TEXT",
                "uploaded_to_cloud": "INTEGER DEFAULT 0",
                "upload_timestamp": "TEXT",
            }
            for col, col_def in expected_cols.items():
                if col not in existing_cols:
                    cursor.execute(f"ALTER TABLE measurements ADD COLUMN {col} {col_def}")
        except Exception:
            # Don't fail init if PRAGMA/ALTER isn't supported; indices below may still fail and reveal issues.
            pass
        
        # Calibration log table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS calibration_log (
                calibration_id TEXT PRIMARY KEY,
                timestamp TEXT NOT NULL,
                operator_id TEXT NOT NULL,
                
                calibrant_material TEXT,
                overall_pass INTEGER NOT NULL,
                
                total_intensity REAL,
                total_intensity_threshold REAL,
                total_intensity_pass INTEGER,
                
                goodness_of_fit REAL,
                goodness_of_fit_threshold REAL,
                goodness_of_fit_pass INTEGER,
                
                snr REAL,
                snr_threshold REAL,
                snr_pass INTEGER,
                
                ring_quality REAL,
                ring_quality_threshold REAL,
                ring_quality_pass INTEGER,
                
                distance_mm REAL,
                beam_center_x REAL,
                beam_center_y REAL,
                wavelength_angstrom REAL,
                
                expires_at TEXT,
                invalidated_at TEXT,
                invalidation_reason TEXT,
                
                formatted_report TEXT
            )
        """)
        
        # gRPC command log table - mirrors hw-server command logging
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS command_logs (
                id TEXT PRIMARY KEY,
                session_id TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                command_type TEXT NOT NULL,
                command_data TEXT NOT NULL,
                hw_server_command_id TEXT,
                user_context TEXT,
                result TEXT NOT NULL,
                execution_time_ms INTEGER NOT NULL,
                error_message TEXT
            )
        """)
        
        # UI command log table - audit trail
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS ui_command_log (
                log_id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                session_id TEXT NOT NULL,
                operator_id TEXT NOT NULL,
                command_type TEXT NOT NULL,
                command_payload TEXT,
                resource_id TEXT,
                result TEXT NOT NULL,
                error_message TEXT,
                ip_address TEXT,
                user_agent TEXT
            )
        """)
        
        # System event log table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS system_event_log (
                event_id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                event_type TEXT NOT NULL,
                severity TEXT NOT NULL,
                component TEXT,
                message TEXT NOT NULL,
                details TEXT,
                operator_id TEXT
            )
        """)
        
        # User sessions table - tracks login/logout
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS user_sessions (
                session_id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                role TEXT NOT NULL,
                login_time TEXT NOT NULL,
                logout_time TEXT,
                last_activity TEXT NOT NULL,
                ip_address TEXT,
                user_agent TEXT
            )
        """)
        
        # Lightweight migration for existing installations: ensure new columns exist on user_sessions
        try:
            cursor.execute("PRAGMA table_info(user_sessions)")
            existing_cols_us = [row[1] for row in cursor.fetchall()]
            # Handle historic typo and missing columns
            if "ip_address" not in existing_cols_us:
                cursor.execute("ALTER TABLE user_sessions ADD COLUMN ip_address TEXT")
            if "user_agent" not in existing_cols_us:
                cursor.execute("ALTER TABLE user_sessions ADD COLUMN user_agent TEXT")
        except Exception:
            pass
        
        # Indexes for common queries
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_patients_mrn 
            ON patients(medical_record_number)
        """)
        
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_measurements_patient 
            ON measurements(patient_id)
        """)
        
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_measurements_timestamp 
            ON measurements(timestamp DESC)
        """)
        
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_measurements_calibration 
            ON measurements(calibration_id)
        """)
        
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_calibration_timestamp 
            ON calibration_log(timestamp DESC)
        """)
        
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_calibration_valid 
            ON calibration_log(overall_pass, expires_at)
        """)
        
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_ui_log_timestamp 
            ON ui_command_log(timestamp DESC)
        """)
        
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_ui_log_operator 
            ON ui_command_log(operator_id)
        """)
        
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_ui_log_resource 
            ON ui_command_log(resource_id)
        """)
        
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_system_log_timestamp 
            ON system_event_log(timestamp DESC)
        """)
        
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_system_log_severity 
            ON system_event_log(severity)
        """)
        
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_sessions_user 
            ON user_sessions(user_id)
        """)
        
        # Initialize schema version if not exists
        cursor.execute("""
            INSERT OR IGNORE INTO schema_version (version, applied_at, description)
            VALUES (1, CURRENT_TIMESTAMP, 'Initial schema with full audit and privacy support')
        """)
        
        conn.commit()
        conn.close()
    
    def create_patient(self, patient: PatientRecord) -> str:
        """Create new patient record.
        
        Args:
            patient: PatientRecord with all patient information
            
        Returns:
            patient_id of created record
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute("""
                INSERT INTO patients (patient_id, first_name, last_name, 
                                      date_of_birth, medical_record_number)
                VALUES (?, ?, ?, ?, ?)
            """, (
                patient.patient_id,
                patient.first_name,
                patient.last_name,
                patient.date_of_birth.isoformat(),
                patient.medical_record_number
            ))
            
            conn.commit()
            return patient.patient_id
            
        except sqlite3.IntegrityError as e:
            conn.rollback()
            raise ValueError(f"Patient record creation failed: {e}")
        finally:
            conn.close()
    
    def get_patient(self, patient_id: str) -> Optional[PatientRecord]:
        """Get patient record by ID."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT patient_id, first_name, last_name, date_of_birth,
                   medical_record_number, created_at, updated_at
            FROM patients WHERE patient_id = ?
        """, (patient_id,))
        
        row = cursor.fetchone()
        conn.close()
        
        if not row:
            return None
        
        return PatientRecord(
            patient_id=row[0],
            first_name=row[1],
            last_name=row[2],
            date_of_birth=datetime.fromisoformat(row[3]),
            medical_record_number=row[4],
            created_at=datetime.fromisoformat(row[5]),
            updated_at=datetime.fromisoformat(row[6])
        )
    
    def find_patient_by_mrn(self, mrn: str) -> Optional[PatientRecord]:
        """Find patient by medical record number."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT patient_id, first_name, last_name, date_of_birth,
                   medical_record_number, created_at, updated_at
            FROM patients WHERE medical_record_number = ?
        """, (mrn,))
        
        row = cursor.fetchone()
        conn.close()
        
        if not row:
            return None
        
        return PatientRecord(
            patient_id=row[0],
            first_name=row[1],
            last_name=row[2],
            date_of_birth=datetime.fromisoformat(row[3]),
            medical_record_number=row[4],
            created_at=datetime.fromisoformat(row[5]),
            updated_at=datetime.fromisoformat(row[6])
        )
    
    def record_measurement(self, measurement: MeasurementRecord):
        """Store measurement record with UUID link to server.
        
        IMPORTANT: The measurement_id UUID is the ONLY piece of data
        shared with the hardware server. Patient information stays local.
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO measurements (
                measurement_id, patient_id, timestamp, operator_id,
                measurement_type, sample_id,
                calibration_id, calibration_timestamp, calibration_valid,
                exposure_duration_ms, beam_intensity_mean, beam_intensity_std,
                snr, detector_temperature, detector_voltage,
                result_data, qc_status, qc_notes, clinical_notes,
                uploaded_to_cloud, upload_timestamp
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            measurement.measurement_id,
            measurement.patient_id,
            measurement.timestamp.isoformat(),
            measurement.operator_id,
            measurement.measurement_type,
            measurement.sample_id,
            measurement.calibration_id,
            measurement.calibration_timestamp.isoformat() if measurement.calibration_timestamp else None,
            1 if measurement.calibration_valid else 0 if measurement.calibration_valid is not None else None,
            measurement.exposure_duration_ms,
            measurement.beam_intensity_mean,
            measurement.beam_intensity_std,
            measurement.snr,
            measurement.detector_temperature,
            measurement.detector_voltage,
            measurement.result_data,
            measurement.qc_status,
            measurement.qc_notes,
            measurement.clinical_notes,
            1 if measurement.uploaded_to_cloud else 0,
            measurement.upload_timestamp.isoformat() if measurement.upload_timestamp else None
        ))
        
        conn.commit()
        conn.close()
    
    def get_measurement(self, measurement_id: str) -> Optional[MeasurementRecord]:
        """Get measurement record by UUID."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT measurement_id, patient_id, timestamp, operator_id,
                   measurement_type, sample_id,
                   calibration_id, calibration_timestamp, calibration_valid,
                   exposure_duration_ms, beam_intensity_mean, beam_intensity_std,
                   snr, detector_temperature, detector_voltage,
                   result_data, qc_status, qc_notes, clinical_notes,
                   uploaded_to_cloud, upload_timestamp
            FROM measurements WHERE measurement_id = ?
        """, (measurement_id,))
        
        row = cursor.fetchone()
        conn.close()
        
        if not row:
            return None
        
        return MeasurementRecord(
            measurement_id=row[0],
            patient_id=row[1],
            timestamp=datetime.fromisoformat(row[2]),
            operator_id=row[3],
            measurement_type=row[4],
            sample_id=row[5],
            calibration_id=row[6],
            calibration_timestamp=datetime.fromisoformat(row[7]) if row[7] else None,
            calibration_valid=bool(row[8]) if row[8] is not None else None,
            exposure_duration_ms=row[9],
            beam_intensity_mean=row[10],
            beam_intensity_std=row[11],
            snr=row[12],
            detector_temperature=row[13],
            detector_voltage=row[14],
            result_data=row[15],
            qc_status=row[16],
            qc_notes=row[17],
            clinical_notes=row[18],
            uploaded_to_cloud=bool(row[19]),
            upload_timestamp=datetime.fromisoformat(row[20]) if row[20] else None
        )
    
    def get_patient_measurements(self, patient_id: str) -> List[MeasurementRecord]:
        """Retrieve all measurements for a patient, ordered by timestamp."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT measurement_id, patient_id, timestamp, operator_id,
                   measurement_type, sample_id,
                   calibration_id, calibration_timestamp, calibration_valid,
                   exposure_duration_ms, beam_intensity_mean, beam_intensity_std,
                   snr, detector_temperature, detector_voltage,
                   result_data, qc_status, qc_notes, clinical_notes,
                   uploaded_to_cloud, upload_timestamp
            FROM measurements 
            WHERE patient_id = ?
            ORDER BY timestamp DESC
        """, (patient_id,))
        
        rows = cursor.fetchall()
        conn.close()
        
        measurements = []
        for row in rows:
            measurements.append(MeasurementRecord(
                measurement_id=row[0],
                patient_id=row[1],
                timestamp=datetime.fromisoformat(row[2]),
                operator_id=row[3],
                measurement_type=row[4],
                sample_id=row[5],
                calibration_id=row[6],
                calibration_timestamp=datetime.fromisoformat(row[7]) if row[7] else None,
                calibration_valid=bool(row[8]) if row[8] is not None else None,
                exposure_duration_ms=row[9],
                beam_intensity_mean=row[10],
                beam_intensity_std=row[11],
                snr=row[12],
                detector_temperature=row[13],
                detector_voltage=row[14],
                result_data=row[15],
                qc_status=row[16],
                qc_notes=row[17],
                clinical_notes=row[18],
                uploaded_to_cloud=bool(row[19]),
                upload_timestamp=datetime.fromisoformat(row[20]) if row[20] else None
            ))
        
        return measurements
    
    def mark_uploaded(self, measurement_id: str, upload_time: datetime):
        """Mark measurement as uploaded to cloud."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            UPDATE measurements 
            SET uploaded_to_cloud = 1, upload_timestamp = ?
            WHERE measurement_id = ?
        """, (upload_time.isoformat(), measurement_id))
        
        conn.commit()
        conn.close()
    
    def record_login(self, user_id: str, session_id: str, role: str, 
                     ip_address: Optional[str] = None, user_agent: Optional[str] = None):
        """Record user login event."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        now = datetime.utcnow().isoformat()
        
        cursor.execute("""
            INSERT INTO user_sessions (session_id, user_id, role, login_time, last_activity, ip_address, user_agent)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (session_id, user_id, role, now, now, ip_address, user_agent))
        
        conn.commit()
        conn.close()
    
    def record_logout(self, session_id: str):
        """Record user logout event."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        now = datetime.utcnow().isoformat()
        
        cursor.execute("""
            UPDATE user_sessions 
            SET logout_time = ?, last_activity = ?
            WHERE session_id = ?
        """, (now, now, session_id))
        
        conn.commit()
        conn.close()
    
    def update_session_activity(self, session_id: str):
        """Update last activity timestamp for session."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        now = datetime.utcnow().isoformat()
        
        cursor.execute("""
            UPDATE user_sessions 
            SET last_activity = ?
            WHERE session_id = ?
        """, (now, session_id))
        
        conn.commit()
        conn.close()
    
    def get_session(self, session_id: str) -> Optional[UserSession]:
        """Get session by ID."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT session_id, user_id, role, login_time, logout_time, last_activity, ip_address, user_agent
            FROM user_sessions WHERE session_id = ?
        """, (session_id,))
        
        row = cursor.fetchone()
        conn.close()
        
        if not row:
            return None
        
        return UserSession(
            session_id=row[0],
            user_id=row[1],
            role=row[2],
            login_time=datetime.fromisoformat(row[3]),
            logout_time=datetime.fromisoformat(row[4]) if row[4] else None,
            last_activity=datetime.fromisoformat(row[5]),
            ip_address=row[6],
            user_agent=row[7]
        )
    
    # Calibration logging methods
    
    def record_calibration(self, calibration: CalibrationRecord):
        """Store calibration record with all QC data."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO calibration_log (
                calibration_id, timestamp, operator_id,
                calibrant_material, overall_pass,
                total_intensity, total_intensity_threshold, total_intensity_pass,
                goodness_of_fit, goodness_of_fit_threshold, goodness_of_fit_pass,
                snr, snr_threshold, snr_pass,
                ring_quality, ring_quality_threshold, ring_quality_pass,
                distance_mm, beam_center_x, beam_center_y, wavelength_angstrom,
                expires_at, invalidated_at, invalidation_reason,
                formatted_report
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            calibration.calibration_id,
            calibration.timestamp.isoformat(),
            calibration.operator_id,
            calibration.calibrant_material,
            1 if calibration.overall_pass else 0,
            calibration.total_intensity,
            calibration.total_intensity_threshold,
            1 if calibration.total_intensity_pass else 0 if calibration.total_intensity_pass is not None else None,
            calibration.goodness_of_fit,
            calibration.goodness_of_fit_threshold,
            1 if calibration.goodness_of_fit_pass else 0 if calibration.goodness_of_fit_pass is not None else None,
            calibration.snr,
            calibration.snr_threshold,
            1 if calibration.snr_pass else 0 if calibration.snr_pass is not None else None,
            calibration.ring_quality,
            calibration.ring_quality_threshold,
            1 if calibration.ring_quality_pass else 0 if calibration.ring_quality_pass is not None else None,
            calibration.distance_mm,
            calibration.beam_center_x,
            calibration.beam_center_y,
            calibration.wavelength_angstrom,
            calibration.expires_at.isoformat() if calibration.expires_at else None,
            calibration.invalidated_at.isoformat() if calibration.invalidated_at else None,
            calibration.invalidation_reason,
            calibration.formatted_report
        ))
        
        conn.commit()
        conn.close()
    
    def get_calibration(self, calibration_id: str) -> Optional[CalibrationRecord]:
        """Get calibration record by ID."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT calibration_id, timestamp, operator_id,
                   calibrant_material, overall_pass,
                   total_intensity, total_intensity_threshold, total_intensity_pass,
                   goodness_of_fit, goodness_of_fit_threshold, goodness_of_fit_pass,
                   snr, snr_threshold, snr_pass,
                   ring_quality, ring_quality_threshold, ring_quality_pass,
                   distance_mm, beam_center_x, beam_center_y, wavelength_angstrom,
                   expires_at, invalidated_at, invalidation_reason,
                   formatted_report
            FROM calibration_log WHERE calibration_id = ?
        """, (calibration_id,))
        
        row = cursor.fetchone()
        conn.close()
        
        if not row:
            return None
        
        return CalibrationRecord(
            calibration_id=row[0],
            timestamp=datetime.fromisoformat(row[1]),
            operator_id=row[2],
            calibrant_material=row[3],
            overall_pass=bool(row[4]),
            total_intensity=row[5],
            total_intensity_threshold=row[6],
            total_intensity_pass=bool(row[7]) if row[7] is not None else None,
            goodness_of_fit=row[8],
            goodness_of_fit_threshold=row[9],
            goodness_of_fit_pass=bool(row[10]) if row[10] is not None else None,
            snr=row[11],
            snr_threshold=row[12],
            snr_pass=bool(row[13]) if row[13] is not None else None,
            ring_quality=row[14],
            ring_quality_threshold=row[15],
            ring_quality_pass=bool(row[16]) if row[16] is not None else None,
            distance_mm=row[17],
            beam_center_x=row[18],
            beam_center_y=row[19],
            wavelength_angstrom=row[20],
            expires_at=datetime.fromisoformat(row[21]) if row[21] else None,
            invalidated_at=datetime.fromisoformat(row[22]) if row[22] else None,
            invalidation_reason=row[23],
            formatted_report=row[24]
        )
    
    def get_current_calibration(self) -> Optional[CalibrationRecord]:
        """Get the most recent valid calibration."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        now = datetime.utcnow().isoformat()
        
        cursor.execute("""
            SELECT calibration_id, timestamp, operator_id,
                   calibrant_material, overall_pass,
                   total_intensity, total_intensity_threshold, total_intensity_pass,
                   goodness_of_fit, goodness_of_fit_threshold, goodness_of_fit_pass,
                   snr, snr_threshold, snr_pass,
                   ring_quality, ring_quality_threshold, ring_quality_pass,
                   distance_mm, beam_center_x, beam_center_y, wavelength_angstrom,
                   expires_at, invalidated_at, invalidation_reason,
                   formatted_report
            FROM calibration_log 
            WHERE overall_pass = 1 
              AND (expires_at IS NULL OR expires_at > ?)
              AND invalidated_at IS NULL
            ORDER BY timestamp DESC
            LIMIT 1
        """, (now,))
        
        row = cursor.fetchone()
        conn.close()
        
        if not row:
            return None
        
        return CalibrationRecord(
            calibration_id=row[0],
            timestamp=datetime.fromisoformat(row[1]),
            operator_id=row[2],
            calibrant_material=row[3],
            overall_pass=bool(row[4]),
            total_intensity=row[5],
            total_intensity_threshold=row[6],
            total_intensity_pass=bool(row[7]) if row[7] is not None else None,
            goodness_of_fit=row[8],
            goodness_of_fit_threshold=row[9],
            goodness_of_fit_pass=bool(row[10]) if row[10] is not None else None,
            snr=row[11],
            snr_threshold=row[12],
            snr_pass=bool(row[13]) if row[13] is not None else None,
            ring_quality=row[14],
            ring_quality_threshold=row[15],
            ring_quality_pass=bool(row[16]) if row[16] is not None else None,
            distance_mm=row[17],
            beam_center_x=row[18],
            beam_center_y=row[19],
            wavelength_angstrom=row[20],
            expires_at=datetime.fromisoformat(row[21]) if row[21] else None,
            invalidated_at=datetime.fromisoformat(row[22]) if row[22] else None,
            invalidation_reason=row[23],
            formatted_report=row[24]
        )
    
    def invalidate_calibration(self, calibration_id: str, reason: str):
        """Manually invalidate a calibration."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        now = datetime.utcnow().isoformat()
        
        cursor.execute("""
            UPDATE calibration_log 
            SET invalidated_at = ?, invalidation_reason = ?
            WHERE calibration_id = ?
        """, (now, reason, calibration_id))
        
        conn.commit()
        conn.close()
    
    def list_calibrations(self, limit: int = 50, since_hours: int | None = 24) -> List[CalibrationRecord]:
        """List recent calibrations, newest first.
        
        Args:
            limit: max number of records to return
            since_hours: if provided, only include calibrations with timestamp >= now - since_hours
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        params = []
        where_clause = ""
        if since_hours is not None:
            from datetime import timedelta
            cutoff = (datetime.utcnow() - timedelta(hours=since_hours)).isoformat()
            where_clause = " WHERE timestamp >= ?"
            params.append(cutoff)
        query = f"""
            SELECT calibration_id, timestamp, operator_id,
                   calibrant_material, overall_pass
            FROM calibration_log
            {where_clause}
            ORDER BY timestamp DESC
            LIMIT ?
        """
        params.append(int(limit))
        cursor.execute(query, tuple(params))
        rows = cursor.fetchall()
        conn.close()
        results: List[CalibrationRecord] = []
        for row in rows:
            # Build a minimal CalibrationRecord; unused fields set to None/defaults
            results.append(CalibrationRecord(
                calibration_id=row[0],
                timestamp=datetime.fromisoformat(row[1]),
                operator_id=row[2],
                calibrant_material=row[3],
                overall_pass=bool(row[4]),
                total_intensity=None,
                total_intensity_threshold=None,
                total_intensity_pass=None,
                goodness_of_fit=None,
                goodness_of_fit_threshold=None,
                goodness_of_fit_pass=None,
                snr=None,
                snr_threshold=None,
                snr_pass=None,
                ring_quality=None,
                ring_quality_threshold=None,
                ring_quality_pass=None,
                distance_mm=None,
                beam_center_x=None,
                beam_center_y=None,
                wavelength_angstrom=None,
                expires_at=None,
                invalidated_at=None,
                invalidation_reason=None,
                formatted_report=None,
            ))
        return results
    
    # gRPC command logging methods
    
    def log_command(self, command_id: str, session_id: str, timestamp: datetime,
                    command_type: str, command_data: dict, hw_server_command_id: Optional[str],
                    user_context: Optional[str], result: str, execution_time_ms: int,
                    error_message: Optional[str] = None):
        """Log a gRPC command execution for orchestrator audit trail.
        
        Args:
            command_id: Orchestrator command ID (UUID)
            session_id: Orchestrator session ID
            timestamp: Command timestamp
            command_type: Service.Command format (e.g., "Acquisition.StartExposure")
            command_data: JSON-serializable dict with command parameters
            hw_server_command_id: Command ID from hw-server (for correlation)
            user_context: User/operator ID
            result: "success" or "failure"
            execution_time_ms: Orchestrator-side execution time
            error_message: Error details if result is "failure"
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        import json
        cursor.execute("""
            INSERT INTO command_logs (
                id, session_id, timestamp, command_type, command_data,
                hw_server_command_id, user_context, result, execution_time_ms, error_message
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            command_id,
            session_id,
            timestamp.isoformat(),
            command_type,
            json.dumps(command_data),
            hw_server_command_id,
            user_context,
            result,
            execution_time_ms,
            error_message
        ))
        
        conn.commit()
        conn.close()
    
    def get_recent_commands(self, limit: int = 100, session_id: Optional[str] = None) -> List[dict]:
        """Get recent gRPC command logs."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        if session_id:
            cursor.execute("""
                SELECT * FROM command_logs
                WHERE session_id = ?
                ORDER BY timestamp DESC
                LIMIT ?
            """, (session_id, limit))
        else:
            cursor.execute("""
                SELECT * FROM command_logs
                ORDER BY timestamp DESC
                LIMIT ?
            """, (limit,))
        
        rows = cursor.fetchall()
        conn.close()
        
        import json
        commands = []
        for row in rows:
            commands.append({
                "id": row[0],
                "session_id": row[1],
                "timestamp": row[2],
                "command_type": row[3],
                "command_data": json.loads(row[4]),
                "hw_server_command_id": row[5],
                "user_context": row[6],
                "result": row[7],
                "execution_time_ms": row[8],
                "error_message": row[9]
            })
        
        return commands
    
    # UI command logging methods
    
    def log_ui_command(self, command: UICommandLog):
        """Log a UI command for audit trail."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO ui_command_log (
                timestamp, session_id, operator_id, command_type,
                command_payload, resource_id, result, error_message,
                ip_address, user_agent
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            command.timestamp.isoformat(),
            command.session_id,
            command.operator_id,
            command.command_type,
            command.command_payload,
            command.resource_id,
            command.result,
            command.error_message,
            command.ip_address,
            command.user_agent
        ))
        
        conn.commit()
        conn.close()
    
    def get_ui_commands(self, operator_id: Optional[str] = None, 
                        limit: int = 100) -> List[UICommandLog]:
        """Retrieve UI command logs, optionally filtered by operator."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        if operator_id:
            cursor.execute("""
                SELECT log_id, timestamp, session_id, operator_id, command_type,
                       command_payload, resource_id, result, error_message,
                       ip_address, user_agent
                FROM ui_command_log
                WHERE operator_id = ?
                ORDER BY timestamp DESC
                LIMIT ?
            """, (operator_id, limit))
        else:
            cursor.execute("""
                SELECT log_id, timestamp, session_id, operator_id, command_type,
                       command_payload, resource_id, result, error_message,
                       ip_address, user_agent
                FROM ui_command_log
                ORDER BY timestamp DESC
                LIMIT ?
            """, (limit,))
        
        rows = cursor.fetchall()
        conn.close()
        
        commands = []
        for row in rows:
            commands.append(UICommandLog(
                log_id=row[0],
                timestamp=datetime.fromisoformat(row[1]),
                session_id=row[2],
                operator_id=row[3],
                command_type=row[4],
                command_payload=row[5],
                resource_id=row[6],
                result=row[7],
                error_message=row[8],
                ip_address=row[9],
                user_agent=row[10]
            ))
        
        return commands
    
    # System event logging methods
    
    def log_system_event(self, event: SystemEventLog):
        """Log a system event."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO system_event_log (
                timestamp, event_type, severity, component,
                message, details, operator_id
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            event.timestamp.isoformat(),
            event.event_type,
            event.severity,
            event.component,
            event.message,
            event.details,
            event.operator_id
        ))
        
        conn.commit()
        conn.close()
    
    def get_system_events(self, severity: Optional[str] = None,
                          component: Optional[str] = None,
                          limit: int = 100) -> List[SystemEventLog]:
        """Retrieve system event logs with optional filters."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        query = """
            SELECT event_id, timestamp, event_type, severity, component,
                   message, details, operator_id
            FROM system_event_log
        """
        params = []
        
        conditions = []
        if severity:
            conditions.append("severity = ?")
            params.append(severity)
        if component:
            conditions.append("component = ?")
            params.append(component)
        
        if conditions:
            query += " WHERE " + " AND ".join(conditions)
        
        query += " ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)
        
        cursor.execute(query, params)
        rows = cursor.fetchall()
        conn.close()
        
        events = []
        for row in rows:
            events.append(SystemEventLog(
                event_id=row[0],
                timestamp=datetime.fromisoformat(row[1]),
                event_type=row[2],
                severity=row[3],
                component=row[4],
                message=row[5],
                details=row[6],
                operator_id=row[7]
            ))
        
        return events
