use omniscan_hw_server::errors::*;

#[test]
fn test_error_codes() {
    // Safety errors
    let error = MedicalDeviceError::SafetyDoorOpen;
    assert_eq!(error.error_code(), "ERR-101");
    
    let error = MedicalDeviceError::EmergencyStopPressed;
    assert_eq!(error.error_code(), "ERR-102");
    
    // Calibration errors
    let error = MedicalDeviceError::CalibrationRequired;
    assert_eq!(error.error_code(), "ERR-201");
    
    // Measurement errors
    let error = MedicalDeviceError::DeviceNotReady { current_state: "LOCKED".to_string() };
    assert_eq!(error.error_code(), "ERR-301");
    
    // Data errors
    let error = MedicalDeviceError::DataStorageFailed { reason: "disk full".to_string() };
    assert_eq!(error.error_code(), "ERR-401");
}

#[test]
fn test_error_severity() {
    // Critical errors
    assert_eq!(
        MedicalDeviceError::AuditLoggingFailed.severity(),
        ErrorSeverity::Critical
    );
    
    // High severity
    assert_eq!(
        MedicalDeviceError::SafetyDoorOpen.severity(),
        ErrorSeverity::High
    );
    
    // Medium severity
    assert_eq!(
        MedicalDeviceError::CalibrationRequired.severity(),
        ErrorSeverity::Medium
    );
    
    // Low severity
    assert_eq!(
        MedicalDeviceError::PhysicalEnableTimeout.severity(),
        ErrorSeverity::Low
    );
}

#[test]
fn test_troubleshooting_steps() {
    let error = MedicalDeviceError::SafetyDoorOpen;
    let steps = error.troubleshooting_steps();
    
    assert!(!steps.is_empty());
    assert!(steps[0].contains("Check"));
    assert!(steps.len() >= 3);
}

#[test]
fn test_requires_support() {
    // Should require support
    assert!(MedicalDeviceError::CalibrationValidationFailed { 
        reason: "test".to_string() 
    }.requires_support());
    
    assert!(MedicalDeviceError::DataStorageFailed { 
        reason: "test".to_string() 
    }.requires_support());
    
    // Should not require support
    assert!(!MedicalDeviceError::PhysicalEnableTimeout.requires_support());
    assert!(!MedicalDeviceError::SafetyDoorOpen.requires_support());
}

#[test]
fn test_error_response_serialization() {
    let error = MedicalDeviceError::CalibrationRequired;
    let response: ErrorResponse = error.into();
    
    assert_eq!(response.error_code, "ERR-201");
    assert!(!response.message.is_empty());
    assert!(!response.troubleshooting_steps.is_empty());
    assert!(!response.timestamp.is_empty());
}

#[test]
fn test_device_not_ready_with_state() {
    let error = MedicalDeviceError::DeviceNotReady { 
        current_state: "CALIBRATION".to_string() 
    };
    
    let steps = error.troubleshooting_steps();
    assert!(steps[0].contains("CALIBRATION"));
}

#[test]
fn test_calibration_expired_message() {
    let error = MedicalDeviceError::CalibrationExpired { hours_ago: 25 };
    let message = error.to_string();
    
    assert!(message.contains("25 hours"));
}
