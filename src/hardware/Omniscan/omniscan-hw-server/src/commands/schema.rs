use serde::{Deserialize, Serialize};
use crate::safety::SafetyState;

/// Complete schema describing a single equipment command
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CommandSchema {
    /// Unique identifier for the command
    pub command_id: String,
    
    /// Human-readable description of what the command does
    pub description: String,
    
    /// Which gRPC service this command belongs to
    pub service: String,
    
    /// Safety requirements that must be met before execution
    pub safety_requirements: SafetyRequirements,
    
    /// Input parameters and their validation rules
    pub parameters: Vec<ParameterSchema>,
    
    /// Step-by-step execution logic
    pub execution: ExecutionSchema,
    
    /// Audit logging level for this command
    pub audit_level: AuditLevel,
    
    /// FDA risk classification
    pub risk_level: RiskLevel,
}

/// Safety requirements for command execution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SafetyRequirements {
    /// Which safety states allow this command
    pub allowed_states: Vec<String>, // Will map to SafetyState enum
    
    /// Does this command require physical enable button press?
    pub requires_enable_button: bool,
    
    /// Must system have valid calibration (within 24h)?
    pub requires_calibration: bool,
    
    /// Does this command require key switch to be ON?
    pub requires_key_switch: bool,
    
    /// Specific interlock checks required
    pub interlock_checks: Vec<InterlockCheck>,
    
    /// Timeout for enable button if required (seconds)
    pub enable_timeout_seconds: Option<u32>,
    
    /// Maximum execution time before safety abort (seconds)
    pub max_execution_time_seconds: Option<u32>,
}

/// Individual interlock requirement
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InterlockCheck {
    /// Name of the interlock signal
    pub name: String,
    
    /// Required value (true/false)
    pub required_value: bool,
    
    /// Human-readable error message if check fails
    pub error_message: String,
}

/// Parameter definition with validation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParameterSchema {
    /// Parameter name (matches protobuf field name)
    pub name: String,
    
    /// Type and constraints
    pub param_type: ParameterType,
    
    /// Is this parameter required?
    pub required: bool,
    
    /// Human-readable description
    pub description: String,
    
    /// Default value if not provided (as JSON string)
    pub default_value: Option<String>,
}

/// Parameter types with built-in validation
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum ParameterType {
    Float {
        unit: String,
        min: f64,
        max: f64,
    },
    Integer {
        min: i64,
        max: i64,
    },
    Boolean,
    String {
        max_length: usize,
        pattern: Option<String>, // Regex pattern
    },
    Enum {
        allowed_values: Vec<String>,
    },
}

/// Execution plan for the command
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutionSchema {
    /// Ordered list of execution steps
    pub steps: Vec<ExecutionStep>,
    
    /// Should we rollback on failure?
    pub rollback_on_failure: bool,
    
    /// Cleanup steps to run on error
    pub error_cleanup: Vec<ExecutionStep>,
}

/// Individual execution step
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum ExecutionStep {
    /// Check a specific interlock
    CheckInterlock {
        name: String,
        required_value: bool,
    },
    
    /// Set a GPIO output pin
    SetGpioOutput {
        pin: String,
        value: bool,
    },
    
    /// Call a device method
    CallDevice {
        device: String,      // "detector", "motion", "gpio", "pdu"
        method: String,      // "start_exposure", "home", etc.
        parameters: Vec<String>, // Parameter names to pass
    },
    
    /// Wait for a condition to be true
    WaitForCondition {
        condition: String,   // "enable_button_pressed", "motion_idle", etc.
        timeout_ms: u32,
        poll_interval_ms: u32,
    },
    
    /// Transition safety state machine
    StateTransition {
        to_state: String,    // Will map to SafetyState
        reason: String,
    },
    
    /// Log an audit event
    LogAudit {
        event_type: String,
        message: String,
    },
    
    /// Emit a state change notification
    NotifyStateChange {
        component: String,
        change_type: String,
    },
    
    /// Conditional execution
    ConditionalBranch {
        condition: String,
        if_true: Vec<ExecutionStep>,
        if_false: Vec<ExecutionStep>,
    },
    
    /// Delay execution
    Delay {
        duration_ms: u32,
    },
}

/// Audit logging level
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AuditLevel {
    /// Minimal logging (query operations)
    Low,
    
    /// Standard logging (most commands)
    Standard,
    
    /// Detailed logging (safety-critical operations)
    High,
    
    /// Full logging with all intermediate states
    Critical,
}

/// FDA risk level classification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RiskLevel {
    /// Low risk - informational/query
    Low,
    
    /// Medium risk - affects device state but not patient
    Medium,
    
    /// High risk - involves radiation or patient exposure
    High,
    
    /// Critical risk - emergency/safety operations
    Critical,
}

impl CommandSchema {
    /// Validate that the schema is well-formed
    pub fn validate(&self) -> Result<(), String> {
        // Check that all parameter names are unique
        let mut param_names = std::collections::HashSet::new();
        for param in &self.parameters {
            if !param_names.insert(&param.name) {
                return Err(format!("Duplicate parameter name: {}", param.name));
            }
        }
        
        // Check that allowed states are valid
        for state in &self.safety_requirements.allowed_states {
            if !Self::is_valid_state(state) {
                return Err(format!("Invalid safety state: {}", state));
            }
        }
        
        Ok(())
    }
    
    fn is_valid_state(state: &str) -> bool {
        matches!(
            state,
            "Locked" | "Initialized" | "Idle" | "WarmingUp" | "Calibrated" 
            | "PendingArmed" | "Running" | "Stopping" | "Safe" 
            | "Calibration" | "Maintenance"
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_schema_validation() {
        let schema = CommandSchema {
            command_id: "test_command".to_string(),
            description: "Test command".to_string(),
            service: "Test".to_string(),
            safety_requirements: SafetyRequirements {
                allowed_states: vec!["Idle".to_string()],
                requires_enable_button: false,
                requires_calibration: false,
                requires_key_switch: true,
                interlock_checks: vec![],
                enable_timeout_seconds: None,
                max_execution_time_seconds: None,
            },
            parameters: vec![],
            execution: ExecutionSchema {
                steps: vec![],
                rollback_on_failure: false,
                error_cleanup: vec![],
            },
            audit_level: AuditLevel::Standard,
            risk_level: RiskLevel::Low,
        };
        
        assert!(schema.validate().is_ok());
    }
    
    #[test]
    fn test_invalid_state() {
        let schema = CommandSchema {
            command_id: "test_command".to_string(),
            description: "Test command".to_string(),
            service: "Test".to_string(),
            safety_requirements: SafetyRequirements {
                allowed_states: vec!["InvalidState".to_string()],
                requires_enable_button: false,
                requires_calibration: false,
                requires_key_switch: true,
                interlock_checks: vec![],
                enable_timeout_seconds: None,
                max_execution_time_seconds: None,
            },
            parameters: vec![],
            execution: ExecutionSchema {
                steps: vec![],
                rollback_on_failure: false,
                error_cleanup: vec![],
            },
            audit_level: AuditLevel::Standard,
            risk_level: RiskLevel::Low,
        };
        
        assert!(schema.validate().is_err());
    }
}
