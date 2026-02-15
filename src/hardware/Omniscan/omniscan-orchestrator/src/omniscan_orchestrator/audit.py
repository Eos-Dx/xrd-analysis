"""
Audit Trail Module for FDA Compliance

Implements append-only audit logging with cryptographic integrity
as required by USER_EXPECTATIONS.md Section 8.
"""

from __future__ import annotations

import hashlib
import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional
import threading


class AuditEvent:
    """Represents a single audit trail event."""
    
    def __init__(
        self,
        event_type: str,
        user_id: str,
        description: str,
        device_id: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        session_id: Optional[str] = None,
    ):
        self.event_id = str(uuid.uuid4())
        self.timestamp = datetime.now(timezone.utc).isoformat()
        self.event_type = event_type
        self.user_id = user_id
        self.description = description
        self.device_id = device_id
        self.details = details or {}
        self.session_id = session_id
        self.previous_hash: Optional[str] = None
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert event to dictionary."""
        return {
            "event_id": self.event_id,
            "timestamp": self.timestamp,
            "event_type": self.event_type,
            "user_id": self.user_id,
            "description": self.description,
            "device_id": self.device_id,
            "details": self.details,
            "session_id": self.session_id,
            "previous_hash": self.previous_hash,
        }
        
    def compute_hash(self) -> str:
        """Compute cryptographic hash of this event."""
        data = json.dumps(self.to_dict(), sort_keys=True)
        return hashlib.sha256(data.encode()).hexdigest()


class AuditLogger:
    """
    Thread-safe audit logger with cryptographic chain-of-custody.
    
    Features:
    - Append-only storage
    - Cryptographic integrity via hash chaining
    - Immediate persistence
    - Thread-safe operations
    """
    
    # Event types as defined in USER_EXPECTATIONS.md Section 8
    EVENT_MEASUREMENT = "measurement"
    EVENT_AUTH = "authentication"
    EVENT_CONFIG_CHANGE = "config_change"
    EVENT_SAFETY_STATUS = "safety_status"
    EVENT_CALIBRATION = "calibration"
    EVENT_SOFTWARE_UPDATE = "software_update"
    EVENT_MAINTENANCE = "maintenance_mode"
    EVENT_ACCESS_DENIED = "access_denied"
    EVENT_SESSION = "session_event"
    EVENT_ERROR = "error_event"
    
    def __init__(self, log_directory: Path, device_id: str):
        self.log_directory = Path(log_directory)
        self.log_directory.mkdir(parents=True, exist_ok=True)
        self.device_id = device_id
        
        # Current log file (rotated daily)
        self.current_log_file = self._get_current_log_file()
        
        # Last hash for chain integrity
        self.last_hash = self._load_last_hash()
        
        # Thread safety
        self._lock = threading.Lock()
        
    def _get_current_log_file(self) -> Path:
        """Get current log file path (date-based)."""
        date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        return self.log_directory / f"audit_{date_str}.jsonl"
        
    def _load_last_hash(self) -> Optional[str]:
        """Load the last hash from the current log file."""
        if not self.current_log_file.exists():
            return None
            
        try:
            with open(self.current_log_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                if lines:
                    last_entry = json.loads(lines[-1])
                    # Reconstruct event and compute its hash
                    event = AuditEvent(
                        event_type=last_entry["event_type"],
                        user_id=last_entry["user_id"],
                        description=last_entry["description"],
                        device_id=last_entry.get("device_id"),
                        details=last_entry.get("details"),
                        session_id=last_entry.get("session_id"),
                    )
                    event.event_id = last_entry["event_id"]
                    event.timestamp = last_entry["timestamp"]
                    event.previous_hash = last_entry.get("previous_hash")
                    return event.compute_hash()
        except Exception:
            pass
            
        return None
        
    def log(
        self,
        event_type: str,
        user_id: str,
        description: str,
        details: Optional[Dict[str, Any]] = None,
        session_id: Optional[str] = None,
    ) -> str:
        """
        Log an audit event with cryptographic integrity.
        
        Returns:
            Event ID for reference
        """
        with self._lock:
            # Create event
            event = AuditEvent(
                event_type=event_type,
                user_id=user_id,
                description=description,
                device_id=self.device_id,
                details=details,
                session_id=session_id,
            )
            
            # Chain to previous event
            event.previous_hash = self.last_hash
            
            # Rotate log file if date changed
            current_log = self._get_current_log_file()
            if current_log != self.current_log_file:
                self.current_log_file = current_log
                self.last_hash = None  # Start new chain
                event.previous_hash = None
            
            # Write to append-only log
            with open(self.current_log_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(event.to_dict()) + '\n')
                f.flush()  # Immediate persistence
                
            # Update last hash
            self.last_hash = event.compute_hash()
            
            return event.event_id
            
    def log_authentication(self, user_id: str, success: bool, method: str, details: Optional[Dict] = None):
        """Log user authentication event."""
        desc = f"Authentication {'successful' if success else 'failed'} via {method}"
        self.log(
            self.EVENT_AUTH,
            user_id,
            desc,
            details={"success": success, "method": method, **(details or {})}
        )
        
    def log_access_denied(self, user_id: str, resource: str, reason: str):
        """Log access denial."""
        self.log(
            self.EVENT_ACCESS_DENIED,
            user_id,
            f"Access denied to {resource}",
            details={"resource": resource, "reason": reason}
        )
        
    def log_config_change(self, user_id: str, key: str, old_value: Any, new_value: Any, session_id: Optional[str] = None):
        """Log configuration change with before/after values."""
        self.log(
            self.EVENT_CONFIG_CHANGE,
            user_id,
            f"Configuration changed: {key}",
            details={"key": key, "old_value": old_value, "new_value": new_value},
            session_id=session_id,
        )
        
    def log_maintenance_mode(self, user_id: str, action: str, session_id: str, details: Optional[Dict] = None):
        """Log maintenance mode entry/exit with explicit tracking."""
        self.log(
            self.EVENT_MAINTENANCE,
            user_id,
            f"Maintenance mode {action}",
            details={"action": action, **(details or {})},
            session_id=session_id,
        )
        
    def log_measurement(self, user_id: str, measurement_id: str, params: Dict, session_id: Optional[str] = None):
        """Log measurement operation with complete parameters."""
        self.log(
            self.EVENT_MEASUREMENT,
            user_id,
            f"Measurement executed: {measurement_id}",
            details={"measurement_id": measurement_id, "parameters": params},
            session_id=session_id,
        )
        
    def log_calibration(self, user_id: str, result: str, details: Dict, session_id: Optional[str] = None):
        """Log calibration event with validation outcome."""
        self.log(
            self.EVENT_CALIBRATION,
            user_id,
            f"Calibration {result}",
            details=details,
            session_id=session_id,
        )
        
    def log_safety_status(self, user_id: str, status_change: str, details: Dict):
        """Log safety system status changes."""
        self.log(
            self.EVENT_SAFETY_STATUS,
            "SYSTEM",  # System-generated events
            f"Safety status: {status_change}",
            details=details,
        )
        
    def log_error(self, user_id: str, error_type: str, error_msg: str, details: Optional[Dict] = None, session_id: Optional[str] = None):
        """Log error events."""
        self.log(
            self.EVENT_ERROR,
            user_id or "SYSTEM",
            f"Error: {error_type}",
            details={"error_message": error_msg, **(details or {})},
            session_id=session_id,
        )
        
    def log_session_event(self, user_id: str, event: str, session_id: str, details: Optional[Dict] = None):
        """Log session lifecycle events."""
        self.log(
            self.EVENT_SESSION,
            user_id,
            f"Session {event}",
            details=details,
            session_id=session_id,
        )
        
    def verify_integrity(self, start_date: Optional[str] = None) -> tuple[bool, Optional[str]]:
        """
        Verify cryptographic integrity of audit trail.
        
        Returns:
            (is_valid, error_message)
        """
        try:
            log_files = sorted(self.log_directory.glob("audit_*.jsonl"))
            
            for log_file in log_files:
                if start_date and log_file.stem.replace("audit_", "") < start_date:
                    continue
                    
                with open(log_file, 'r', encoding='utf-8') as f:
                    previous_hash = None
                    
                    for line_num, line in enumerate(f, 1):
                        try:
                            entry = json.loads(line)
                            
                            # Verify hash chain
                            if entry.get("previous_hash") != previous_hash:
                                return False, f"Hash chain broken at {log_file}:{line_num}"
                                
                            # Reconstruct and compute hash
                            event = AuditEvent(
                                event_type=entry["event_type"],
                                user_id=entry["user_id"],
                                description=entry["description"],
                                device_id=entry.get("device_id"),
                                details=entry.get("details"),
                                session_id=entry.get("session_id"),
                            )
                            event.event_id = entry["event_id"]
                            event.timestamp = entry["timestamp"]
                            event.previous_hash = entry.get("previous_hash")
                            
                            previous_hash = event.compute_hash()
                            
                        except json.JSONDecodeError:
                            return False, f"Invalid JSON at {log_file}:{line_num}"
                            
            return True, None
            
        except Exception as e:
            return False, f"Verification error: {str(e)}"
            
    def export_audit_trail(self, start_date: str, end_date: str, output_path: Path, format: str = "json"):
        """
        Export audit trail for regulatory review.
        
        Supports formats: json, csv
        """
        events = []
        
        log_files = sorted(self.log_directory.glob("audit_*.jsonl"))
        
        for log_file in log_files:
            file_date = log_file.stem.replace("audit_", "")
            if start_date <= file_date <= end_date:
                with open(log_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        events.append(json.loads(line))
                        
        if format == "json":
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(events, f, indent=2)
                
        elif format == "csv":
            import csv
            with open(output_path, 'w', newline='', encoding='utf-8') as f:
                if events:
                    writer = csv.DictWriter(f, fieldnames=events[0].keys())
                    writer.writeheader()
                    writer.writerows(events)


# Global audit logger instance
_audit_logger: Optional[AuditLogger] = None


def initialize_audit_logger(log_directory: Path, device_id: str) -> AuditLogger:
    """Initialize the global audit logger."""
    global _audit_logger
    _audit_logger = AuditLogger(log_directory, device_id)
    return _audit_logger


def get_audit_logger() -> Optional[AuditLogger]:
    """Get the global audit logger instance."""
    return _audit_logger
