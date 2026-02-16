//! User-friendly error handling for medical device operations
//! 
//! Implements requirements from USER_EXPECTATIONS.md Section 5:
//! - Plain language error descriptions
//! - Numbered troubleshooting steps
//! - Unique error reference numbers
//! - Clear indication when to contact support

use thiserror::Error;

/// Medical device error with user-friendly messaging
#[derive(Error, Debug, Clone)]
pub enum MedicalDeviceError {
    // Safety Errors (ERR-100 series)
    #[error("[ERR-101] Safety door is open. Cannot operate device.")]
    SafetyDoorOpen,
    
    #[error("[ERR-102] Emergency stop button is pressed. Release to continue.")]
    EmergencyStopPressed,
    
    #[error("[ERR-103] Key switch is not in operate position.")]
    KeySwitchNotInOperate,
    
    #[error("[ERR-104] X-ray beam intensity out of safe limits.")]
    BeamIntensityOutOfLimits,
    
    #[error("[ERR-105] Device over-temperature detected.")]
    OverTemperature,
    
    // Calibration Errors (ERR-200 series)
    #[error("[ERR-201] Daily calibration required. Device is locked.")]
    CalibrationRequired,
    
    #[error("[ERR-202] Calibration failed validation. Contact service.")]
    CalibrationValidationFailed { reason: String },
    
    #[error("[ERR-203] Calibration expired. Last calibration was {hours_ago} hours ago.")]
    CalibrationExpired { hours_ago: u32 },
    
    // Measurement Errors (ERR-300 series)
    #[error("[ERR-301] Cannot start measurement. Device not ready.")]
    DeviceNotReady { current_state: String },
    
    #[error("[ERR-302] Physical enable button not pressed within timeout.")]
    PhysicalEnableTimeout,
    
    #[error("[ERR-303] Measurement aborted due to safety interlock violation.")]
    MeasurementAbortedInterlock,
    
    #[error("[ERR-304] Detector communication error.")]
    DetectorCommunicationError { details: String },
    
    // Data Integrity Errors (ERR-400 series)
    #[error("[ERR-401] Failed to save measurement data. Contact IT support.")]
    DataStorageFailed { reason: String },
    
    #[error("[ERR-402] Data encryption failed. Contact IT support.")]
    EncryptionFailed,
    
    #[error("[ERR-403] Audit logging failure. Operation cannot proceed.")]
    AuditLoggingFailed,
    
    // Hardware Errors (ERR-500 series)
    #[error("[ERR-501] Motion controller communication error.")]
    MotionControllerError { details: String },
    
    #[error("[ERR-502] GPIO device not responding.")]
    GpioDeviceError,
    
    #[error("[ERR-503] Power distribution unit communication error.")]
    PduCommunicationError,
    
    // Authentication/Authorization Errors (ERR-600 series)
    #[error("[ERR-601] User not authorized for this operation.")]
    Unauthorized { required_role: String },
    
    #[error("[ERR-602] Session expired. Please log in again.")]
    SessionExpired,
    
    #[error("[ERR-603] Invalid credentials.")]
    InvalidCredentials,
    
    // System Errors (ERR-700 series)
    #[error("[ERR-701] System initialization failed.")]
    SystemInitializationFailed { component: String },
    
    #[error("[ERR-702] Configuration error. Contact IT support.")]
    ConfigurationError { details: String },
    
    #[error("[ERR-703] Network connectivity issue.")]
    NetworkError,
}

impl MedicalDeviceError {
    /// Get the error reference number for support coordination
    pub fn error_code(&self) -> String {
        let msg = self.to_string();
        if let Some(start) = msg.find("[ERR-") {
            if let Some(end) = msg[start..].find(']') {
                return msg[start+1..start+end].to_string();
            }
        }
        "ERR-000".to_string()
    }
    
    /// Get troubleshooting steps for operator
    pub fn troubleshooting_steps(&self) -> Vec<String> {
        match self {
            MedicalDeviceError::SafetyDoorOpen => vec![
                "1. Check that the safety door is fully closed".to_string(),
                "2. Listen for the door latch click".to_string(),
                "3. Check door interlock sensor LED (should be green)".to_string(),
                "4. If door is closed but error persists, contact service".to_string(),
            ],
            MedicalDeviceError::EmergencyStopPressed => vec![
                "1. Rotate emergency stop button clockwise to release".to_string(),
                "2. Verify red E-stop indicator is off".to_string(),
                "3. If button is released but error persists, contact service".to_string(),
            ],
            MedicalDeviceError::CalibrationRequired => vec![
                "1. Run daily calibration procedure from main menu".to_string(),
                "2. Place calibration standard in sample holder".to_string(),
                "3. Follow on-screen calibration instructions".to_string(),
                "4. Wait for calibration completion (approximately 10 minutes)".to_string(),
            ],
            MedicalDeviceError::DeviceNotReady { current_state } => vec![
                format!("Device is currently in '{}' state", current_state),
                "1. Check system status display".to_string(),
                "2. Wait for any ongoing operations to complete".to_string(),
                "3. Verify all safety interlocks are satisfied".to_string(),
                "4. If device remains not ready, contact service".to_string(),
            ],
            MedicalDeviceError::PhysicalEnableTimeout => vec![
                "1. Press the green enable button within 30 seconds".to_string(),
                "2. Restart measurement if timeout occurred".to_string(),
                "3. Ensure you are ready before initiating measurement".to_string(),
            ],
            MedicalDeviceError::DataStorageFailed { .. } => vec![
                "1. Check available disk space".to_string(),
                "2. Verify network connectivity".to_string(),
                "3. Contact IT support immediately".to_string(),
                "4. Do not proceed with measurements until resolved".to_string(),
            ],
            _ => vec![
                "1. Note the error code for support reference".to_string(),
                "2. Contact technical support with error code".to_string(),
            ],
        }
    }
    
    /// Whether user should contact support
    pub fn requires_support(&self) -> bool {
        matches!(self,
            MedicalDeviceError::CalibrationValidationFailed { .. } |
            MedicalDeviceError::DataStorageFailed { .. } |
            MedicalDeviceError::EncryptionFailed |
            MedicalDeviceError::AuditLoggingFailed |
            MedicalDeviceError::GpioDeviceError |
            MedicalDeviceError::PduCommunicationError |
            MedicalDeviceError::SystemInitializationFailed { .. } |
            MedicalDeviceError::ConfigurationError { .. }
        )
    }
    
    /// Severity level for logging and alerting
    pub fn severity(&self) -> ErrorSeverity {
        match self {
            // Critical - prevents all operations
            MedicalDeviceError::AuditLoggingFailed |
            MedicalDeviceError::EncryptionFailed |
            MedicalDeviceError::SystemInitializationFailed { .. } => ErrorSeverity::Critical,
            
            // High - safety or data integrity issues
            MedicalDeviceError::SafetyDoorOpen |
            MedicalDeviceError::EmergencyStopPressed |
            MedicalDeviceError::BeamIntensityOutOfLimits |
            MedicalDeviceError::DataStorageFailed { .. } => ErrorSeverity::High,
            
            // Medium - operational issues
            MedicalDeviceError::CalibrationRequired |
            MedicalDeviceError::CalibrationExpired { .. } |
            MedicalDeviceError::DeviceNotReady { .. } |
            MedicalDeviceError::DetectorCommunicationError { .. } => ErrorSeverity::Medium,
            
            // Low - user errors or timeouts
            MedicalDeviceError::PhysicalEnableTimeout |
            MedicalDeviceError::Unauthorized { .. } |
            MedicalDeviceError::SessionExpired => ErrorSeverity::Low,
            
            _ => ErrorSeverity::Medium,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ErrorSeverity {
    Low,
    Medium,
    High,
    Critical,
}

/// User-friendly error response for API calls
#[derive(Debug, Clone, serde::Serialize)]
pub struct ErrorResponse {
    pub error_code: String,
    pub message: String,
    pub severity: String,
    pub troubleshooting_steps: Vec<String>,
    pub requires_support: bool,
    pub timestamp: String,
}

impl From<MedicalDeviceError> for ErrorResponse {
    fn from(error: MedicalDeviceError) -> Self {
        Self {
            error_code: error.error_code().to_string(),
            message: error.to_string(),
            severity: format!("{:?}", error.severity()),
            troubleshooting_steps: error.troubleshooting_steps(),
            requires_support: error.requires_support(),
            timestamp: chrono::Utc::now().to_rfc3339(),
        }
    }
}