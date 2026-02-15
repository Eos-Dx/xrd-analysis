"""
Safety-First Error Handling Module

Implements clear error messages with actionable steps as required by
USER_EXPECTATIONS.md Section 5.

All errors include:
- Plain language description for operators
- Numbered troubleshooting steps
- Unique error reference numbers
- Technical details for maintenance
"""

from __future__ import annotations

from enum import Enum
from typing import Optional, List
from dataclasses import dataclass


class ErrorSeverity(Enum):
    """Error severity levels."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"  # Requires immediate attention, affects safety


class ErrorCategory(Enum):
    """Error categories for classification."""
    SAFETY = "safety"
    HARDWARE = "hardware"
    CALIBRATION = "calibration"
    MEASUREMENT = "measurement"
    CONFIGURATION = "configuration"
    AUTHENTICATION = "authentication"
    NETWORK = "network"
    DATA_STORAGE = "data_storage"


@dataclass
class ErrorCode:
    """Structured error code with context."""
    code: str  # e.g., "SAF-001"
    category: ErrorCategory
    severity: ErrorSeverity
    title: str
    description: str
    troubleshooting_steps: List[str]
    contact_support: bool = False
    auto_safe_state: bool = False  # Automatically transition to safe state
    
    def format_for_operator(self) -> str:
        """Format error message for clinical operators (plain language)."""
        lines = [
            f"⚠️  {self.title}",
            f"",
            f"Error Code: {self.code}",
            f"",
            f"{self.description}",
            f"",
        ]
        
        if self.troubleshooting_steps:
            lines.append("What to do:")
            for i, step in enumerate(self.troubleshooting_steps, 1):
                lines.append(f"  {i}. {step}")
            lines.append("")
            
        if self.contact_support:
            lines.append("📞 If problem persists, contact support with error code " + self.code)
            
        return "\n".join(lines)
        
    def format_for_maintenance(self, technical_details: Optional[dict] = None) -> str:
        """Format error message for maintenance engineers (with technical details)."""
        lines = [
            f"ERROR {self.code}: {self.title}",
            f"Category: {self.category.value}",
            f"Severity: {self.severity.value}",
            f"",
            f"Description: {self.description}",
            f"",
        ]
        
        if technical_details:
            lines.append("Technical Details:")
            for key, value in technical_details.items():
                lines.append(f"  {key}: {value}")
            lines.append("")
            
        if self.troubleshooting_steps:
            lines.append("Troubleshooting:")
            for i, step in enumerate(self.troubleshooting_steps, 1):
                lines.append(f"  {i}. {step}")
                
        return "\n".join(lines)


# Error code registry as per USER_EXPECTATIONS.md
ERROR_CODES = {
    # Safety errors (SAF-xxx) - CRITICAL
    "SAF-001": ErrorCode(
        code="SAF-001",
        category=ErrorCategory.SAFETY,
        severity=ErrorSeverity.CRITICAL,
        title="Safety Door Open",
        description="The safety door is not fully closed. X-ray beam cannot be activated.",
        troubleshooting_steps=[
            "Ensure the safety door is completely closed",
            "Check for obstructions preventing door closure",
            "Verify door latch is engaged (listen for click)",
            "If door is closed but error persists, contact maintenance"
        ],
        contact_support=True,
        auto_safe_state=True,
    ),
    
    "SAF-002": ErrorCode(
        code="SAF-002",
        category=ErrorCategory.SAFETY,
        severity=ErrorSeverity.CRITICAL,
        title="Emergency Stop Activated",
        description="The emergency stop button has been pressed. All operations are halted.",
        troubleshooting_steps=[
            "Identify reason for emergency stop activation",
            "Reset emergency stop button by twisting clockwise",
            "Verify all personnel are clear of equipment",
            "Reset system and check all safety interlocks"
        ],
        contact_support=False,
        auto_safe_state=True,
    ),
    
    "SAF-003": ErrorCode(
        code="SAF-003",
        category=ErrorCategory.SAFETY,
        severity=ErrorSeverity.CRITICAL,
        title="Safety Interlock Fault",
        description="A safety interlock has failed. System cannot operate.",
        troubleshooting_steps=[
            "Do not attempt to operate the system",
            "Document any unusual events before fault",
            "Contact maintenance immediately"
        ],
        contact_support=True,
        auto_safe_state=True,
    ),
    
    # Calibration errors (CAL-xxx)
    "CAL-001": ErrorCode(
        code="CAL-001",
        category=ErrorCategory.CALIBRATION,
        severity=ErrorSeverity.ERROR,
        title="Calibration Expired",
        description="Daily calibration has not been performed within 24 hours. Measurements are blocked.",
        troubleshooting_steps=[
            "Perform daily calibration routine",
            "Ensure calibration standard is properly positioned",
            "Follow on-screen calibration instructions",
            "Verify calibration passes acceptance criteria"
        ],
        contact_support=False,
        auto_safe_state=False,
    ),
    
    "CAL-002": ErrorCode(
        code="CAL-002",
        category=ErrorCategory.CALIBRATION,
        severity=ErrorSeverity.WARNING,
        title="Calibration Failed",
        description="Calibration did not meet acceptance criteria.",
        troubleshooting_steps=[
            "Verify calibration standard is correctly positioned",
            "Check that sample chamber is clean and free of debris",
            "Ensure X-ray tube has warmed up (run 1-2 test exposures)",
            "Retry calibration",
            "If still failing, contact maintenance"
        ],
        contact_support=True,
        auto_safe_state=False,
    ),
    
    # Hardware errors (HW-xxx)
    "HW-001": ErrorCode(
        code="HW-001",
        category=ErrorCategory.HARDWARE,
        severity=ErrorSeverity.ERROR,
        title="X-ray Source Not Responding",
        description="The X-ray tube is not responding to control commands.",
        troubleshooting_steps=[
            "Check X-ray generator power status",
            "Verify all cables are securely connected",
            "Power cycle the X-ray generator (wait 30 seconds)",
            "If problem persists, contact maintenance"
        ],
        contact_support=True,
        auto_safe_state=True,
    ),
    
    "HW-002": ErrorCode(
        code="HW-002",
        category=ErrorCategory.HARDWARE,
        severity=ErrorSeverity.ERROR,
        title="Detector Communication Lost",
        description="Cannot communicate with the X-ray detector.",
        troubleshooting_steps=[
            "Check detector power and connections",
            "Verify USB/Ethernet cable is connected",
            "Restart detector hardware",
            "Contact maintenance if issue persists"
        ],
        contact_support=True,
        auto_safe_state=False,
    ),
    
    # Measurement errors (MEAS-xxx)
    "MEAS-001": ErrorCode(
        code="MEAS-001",
        category=ErrorCategory.MEASUREMENT,
        severity=ErrorSeverity.WARNING,
        title="Beam Intensity Out of Range",
        description="X-ray beam intensity deviated from expected range during measurement.",
        troubleshooting_steps=[
            "Data quality may be compromised",
            "Consider re-running the measurement",
            "Check X-ray tube warm-up status",
            "If recurring, schedule maintenance"
        ],
        contact_support=False,
        auto_safe_state=False,
    ),
    
    "MEAS-002": ErrorCode(
        code="MEAS-002",
        category=ErrorCategory.MEASUREMENT,
        severity=ErrorSeverity.ERROR,
        title="Measurement Interrupted",
        description="The measurement was interrupted before completion.",
        troubleshooting_steps=[
            "Review reason for interruption",
            "Ensure sample is properly positioned",
            "Check for safety interlock triggers",
            "Retry measurement"
        ],
        contact_support=False,
        auto_safe_state=False,
    ),
    
    # Configuration errors (CFG-xxx)
    "CFG-001": ErrorCode(
        code="CFG-001",
        category=ErrorCategory.CONFIGURATION,
        severity=ErrorSeverity.ERROR,
        title="Invalid Configuration",
        description="System configuration contains invalid parameters.",
        troubleshooting_steps=[
            "Review recent configuration changes",
            "Restore previous working configuration",
            "Contact administrator for configuration assistance"
        ],
        contact_support=True,
        auto_safe_state=False,
    ),
    
    # Authentication errors (AUTH-xxx)
    "AUTH-001": ErrorCode(
        code="AUTH-001",
        category=ErrorCategory.AUTHENTICATION,
        severity=ErrorSeverity.ERROR,
        title="Authentication Failed",
        description="User authentication failed. Access denied.",
        troubleshooting_steps=[
            "Verify certificate is valid and not expired",
            "Check certificate is from authorized Certificate Authority",
            "Ensure certificate matches this device UUID",
            "Contact administrator if problem persists"
        ],
        contact_support=True,
        auto_safe_state=False,
    ),
    
    "AUTH-002": ErrorCode(
        code="AUTH-002",
        category=ErrorCategory.AUTHENTICATION,
        severity=ErrorSeverity.ERROR,
        title="Insufficient Permissions",
        description="User does not have permission for this operation.",
        troubleshooting_steps=[
            "Contact administrator to verify your role and permissions",
            "Ensure you are using the correct user certificate",
            "Review operation requirements in user manual"
        ],
        contact_support=True,
        auto_safe_state=False,
    ),
    
    # Data storage errors (DATA-xxx)
    "DATA-001": ErrorCode(
        code="DATA-001",
        category=ErrorCategory.DATA_STORAGE,
        severity=ErrorSeverity.CRITICAL,
        title="Data Storage Full",
        description="Local data storage is critically low. Measurements may fail.",
        troubleshooting_steps=[
            "Contact IT administrator immediately",
            "Do not perform measurements until storage is cleared",
            "Archive or delete old data if authorized",
            "Ensure cloud sync is functioning"
        ],
        contact_support=True,
        auto_safe_state=False,
    ),
    
    "DATA-002": ErrorCode(
        code="DATA-002",
        category=ErrorCategory.DATA_STORAGE,
        severity=ErrorSeverity.ERROR,
        title="Data Write Failed",
        description="Failed to save measurement data to storage.",
        troubleshooting_steps=[
            "Check disk space availability",
            "Verify storage permissions",
            "Contact IT administrator",
            "Do not perform measurements until resolved"
        ],
        contact_support=True,
        auto_safe_state=True,
    ),
    
    # Network errors (NET-xxx)
    "NET-001": ErrorCode(
        code="NET-001",
        category=ErrorCategory.NETWORK,
        severity=ErrorSeverity.WARNING,
        title="Cloud Connection Lost",
        description="Cannot connect to cloud services. Operating in offline mode.",
        troubleshooting_steps=[
            "Measurements can continue - data is stored locally",
            "Check network cable connection",
            "Verify firewall settings allow cloud access",
            "Data will sync automatically when connection restores"
        ],
        contact_support=False,
        auto_safe_state=False,
    ),
}


class OmniscanError(Exception):
    """Base exception for Omniscan orchestrator."""
    
    def __init__(
        self,
        error_code: ErrorCode,
        technical_details: Optional[dict] = None,
        user_message: Optional[str] = None,
    ):
        self.error_code = error_code
        self.technical_details = technical_details or {}
        self.user_message = user_message or error_code.format_for_operator()
        super().__init__(self.user_message)
        
    def should_transition_safe_state(self) -> bool:
        """Check if this error requires automatic safe state transition."""
        return self.error_code.auto_safe_state
        
    def get_error_ref(self) -> str:
        """Get error reference number for support."""
        return self.error_code.code


def get_error(code: str, **technical_details) -> OmniscanError:
    """
    Get an OmniscanError by code with technical details.
    
    Example:
        raise get_error("SAF-001", door_sensor_value=0, timestamp="2025-10-24T12:00:00Z")
    """
    error_code = ERROR_CODES.get(code)
    if not error_code:
        raise ValueError(f"Unknown error code: {code}")
        
    return OmniscanError(error_code, technical_details=technical_details)
