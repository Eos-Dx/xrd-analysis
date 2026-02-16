use std::sync::Arc;
use tokio::sync::RwLock;
use chrono::{DateTime, Utc};
use tracing::{info, warn, error};
use anyhow::Result;

use crate::audit::{AuditLogger, CommandLog, CommandResult};

/// Safety state machine as defined in SYS_OMNI-SERVER_003
#[derive(Debug, Clone, PartialEq)]
pub enum SafetyState {
    Locked,          // Key switch OFF - no operations allowed
    Initialized,     // Key switch ON, awaiting login
    Idle,            // Ready for operations
    WarmingUp,       // X-ray source heat-up in progress
    Calibrated,      // Valid calibration within 24 hours
    PendingArmed,    // Waiting for physical enable
    Running,         // Active measurement or operation
    Stopping,        // Controlled shutdown in progress
    Safe,            // Emergency safe state
    Calibration,     // Calibration measurement in progress
    Maintenance,     // Maintenance mode active
}

impl SafetyState {
    pub fn as_str(&self) -> &'static str {
        match self {
            SafetyState::Locked => "LOCKED",
            SafetyState::Initialized => "INITIALIZED",
            SafetyState::Idle => "IDLE",
            SafetyState::WarmingUp => "WARMING_UP",
            SafetyState::Calibrated => "CALIBRATED",
            SafetyState::PendingArmed => "PENDING_ARMED",
            SafetyState::Running => "RUNNING",
            SafetyState::Stopping => "STOPPING",
            SafetyState::Safe => "SAFE",
            SafetyState::Calibration => "CALIBRATION",
            SafetyState::Maintenance => "MAINTENANCE",
        }
    }
}

/// Interlock status as defined in SYS_OMNI-SERVER_002
#[derive(Debug, Clone)]
pub struct InterlockStatus {
    pub key_switch: bool,           // Key switch in operate position
    pub enable_button: bool,        // Enable button pressed
    pub emergency_stop: bool,       // E-stop not pressed
    pub door_closed: bool,          // Safety door closed
    pub beam_watchdog: bool,        // Beam intensity within limits
    pub over_temperature: Option<bool>, // Optional over-temperature sensor
    pub overall_safe: bool,         // All required interlocks satisfied
    pub violation_reason: Option<String>, // Reason if not safe
    pub last_check: DateTime<Utc>,
}

impl InterlockStatus {
    pub fn new() -> Self {
        Self {
            key_switch: false,
            enable_button: false,
            emergency_stop: false,
            door_closed: false,
            beam_watchdog: true,  // Default to safe
            over_temperature: Some(true), // Default to safe
            overall_safe: false,
            violation_reason: Some("Interlocks not initialized".to_string()),
            last_check: Utc::now(),
        }
    }

    /// Evaluate overall safety based on individual interlocks
    pub fn evaluate_safety(&mut self) {
        let mut violations = Vec::new();

        if !self.key_switch {
            violations.push("Key switch not in operate position");
        }
        if !self.emergency_stop {
            violations.push("Emergency stop pressed");
        }
        if !self.door_closed {
            violations.push("Safety door open");
        }
        if !self.beam_watchdog {
            violations.push("Beam intensity out of limits");
        }
        if let Some(false) = self.over_temperature {
            violations.push("Over-temperature condition");
        }

        self.overall_safe = violations.is_empty();
        self.violation_reason = if violations.is_empty() {
            None
        } else {
            Some(violations.join(", "))
        };
        self.last_check = Utc::now();
    }
}

/// State transition reasons for audit trail
#[derive(Debug, Clone)]
pub enum StateTransitionReason {
    UserCommand(String),      // User-initiated command
    InterlockViolation(String), // Interlock failure
    SystemTimeout,            // Timeout expired
    CalibrationRequired,      // Calibration needed
    MaintenanceMode,          // Maintenance activated
    EmergencyAbort,          // Emergency abort
    SystemStartup,           // System initialization
}

impl StateTransitionReason {
    pub fn as_str(&self) -> String {
        match self {
            StateTransitionReason::UserCommand(cmd) => format!("User command: {}", cmd),
            StateTransitionReason::InterlockViolation(reason) => format!("Interlock violation: {}", reason),
            StateTransitionReason::SystemTimeout => "System timeout".to_string(),
            StateTransitionReason::CalibrationRequired => "Calibration required".to_string(),
            StateTransitionReason::MaintenanceMode => "Maintenance mode activated".to_string(),
            StateTransitionReason::EmergencyAbort => "Emergency abort".to_string(),
            StateTransitionReason::SystemStartup => "System startup".to_string(),
        }
    }
}

/// Safety state machine for medical device control
pub struct SafetyStateMachine {
    current_state: Arc<RwLock<SafetyState>>,
    interlock_status: Arc<RwLock<InterlockStatus>>,
    audit_logger: Arc<AuditLogger>,
    pending_armed_timeout: Option<DateTime<Utc>>,
    last_calibration: Option<DateTime<Utc>>,
    last_calibration_qc_json: Option<String>,  // Store QC report as JSON string
    calibration_required_interval: chrono::Duration,
}

impl SafetyStateMachine {
    pub fn new(audit_logger: Arc<AuditLogger>) -> Self {
        Self {
            current_state: Arc::new(RwLock::new(SafetyState::Locked)),
            interlock_status: Arc::new(RwLock::new(InterlockStatus::new())),
            audit_logger,
            pending_armed_timeout: None,
            last_calibration: None,
            last_calibration_qc_json: None,
            calibration_required_interval: chrono::Duration::hours(24), // SYS_OMNI-SERVER_004
        }
    }

    /// Initialize the safety system and transition to IDLE if safe
    pub async fn initialize(&mut self) -> Result<()> {
        info!("Initializing safety state machine");

        // Check initial interlock status
        let mut interlocks = self.interlock_status.write().await;
        interlocks.evaluate_safety();
        
        let initial_state = if interlocks.overall_safe {
            SafetyState::Idle
        } else {
            SafetyState::Safe
        };

        drop(interlocks);

        self.transition_to(initial_state, StateTransitionReason::SystemStartup).await?;
        
        info!("Safety state machine initialized");
        Ok(())
    }

    /// Get current state
    pub async fn get_current_state(&self) -> SafetyState {
        self.current_state.read().await.clone()
    }

    /// Get current interlock status
    pub async fn get_interlock_status(&self) -> InterlockStatus {
        self.interlock_status.read().await.clone()
    }

    /// Update interlock status (called by GPIO/hardware monitoring)
    pub async fn update_interlocks(&mut self, new_status: InterlockStatus) -> Result<()> {
        let old_overall_safe = {
            let interlocks = self.interlock_status.read().await;
            interlocks.overall_safe
        };

        {
            let mut interlocks = self.interlock_status.write().await;
            *interlocks = new_status;
            interlocks.evaluate_safety();
        }

        let new_overall_safe = {
            let interlocks = self.interlock_status.read().await;
            interlocks.overall_safe
        };

        // Handle interlock state changes
        if old_overall_safe && !new_overall_safe {
            // Interlocks violated - transition to SAFE
            let violation_reason = {
                let interlocks = self.interlock_status.read().await;
                interlocks.violation_reason.clone().unwrap_or_default()
            };
            
            warn!("Interlock violation detected: {}", violation_reason);
            self.transition_to(
                SafetyState::Safe, 
                StateTransitionReason::InterlockViolation(violation_reason)
            ).await?;
        }

        Ok(())
    }

    /// Attempt to start a measurement (user command)
    pub async fn start_measurement(&mut self, command_id: String, _user: String) -> Result<bool> {
        let current_state = self.get_current_state().await;
        let interlocks = self.get_interlock_status().await;

        // Check calibration requirement (SYS_OMNI-SERVER_004)
        if !self.is_calibration_valid() {
            warn!("Measurement blocked: calibration required");
            self.transition_to(SafetyState::Locked, StateTransitionReason::CalibrationRequired).await?;
            return Ok(false);
        }

        match current_state {
            SafetyState::Idle => {
                if !interlocks.overall_safe {
                    warn!("Measurement blocked: interlocks not satisfied");
                    self.transition_to(
                        SafetyState::Safe, 
                        StateTransitionReason::InterlockViolation(
                            interlocks.violation_reason.unwrap_or_default()
                        )
                    ).await?;
                    return Ok(false);
                }

                // Check if physical enable is required
                if self.requires_physical_enable() {
                    info!("Physical enable required for measurement");
                    self.transition_to(
                        SafetyState::PendingArmed,
                        StateTransitionReason::UserCommand(format!("start_measurement:{}", command_id))
                    ).await?;
                    
                    // Set timeout for physical enable
                    self.pending_armed_timeout = Some(Utc::now() + chrono::Duration::seconds(30));
                } else {
                    // Direct transition to running
                    self.transition_to(
                        SafetyState::Running,
                        StateTransitionReason::UserCommand(format!("start_measurement:{}", command_id))
                    ).await?;
                }
                Ok(true)
            }
            _ => {
                warn!("Cannot start measurement from state: {}", current_state.as_str());
                Ok(false)
            }
        }
    }

    /// Handle physical enable button press
    pub async fn physical_enable_pressed(&mut self) -> Result<()> {
        let current_state = self.get_current_state().await;
        
        if current_state == SafetyState::PendingArmed {
            // Check if still within timeout window
            if let Some(timeout) = self.pending_armed_timeout {
                if Utc::now() <= timeout {
                    info!("Physical enable confirmed within timeout window");
                    self.transition_to(SafetyState::Running, StateTransitionReason::UserCommand("physical_enable".to_string())).await?;
                    self.pending_armed_timeout = None;
                } else {
                    warn!("Physical enable timeout expired");
                    self.transition_to(SafetyState::Idle, StateTransitionReason::SystemTimeout).await?;
                    self.pending_armed_timeout = None;
                }
            }
        }

        Ok(())
    }

    /// Abort operation (emergency)
    pub async fn abort(&mut self, command_id: String) -> Result<()> {
        info!("Emergency abort commanded: {}", command_id);
        self.transition_to(SafetyState::Safe, StateTransitionReason::EmergencyAbort).await?;
        self.pending_armed_timeout = None;
        Ok(())
    }

    /// Stop operation (normal)
    pub async fn stop(&mut self, command_id: String) -> Result<()> {
        let current_state = self.get_current_state().await;
        
        match current_state {
            SafetyState::Running | SafetyState::PendingArmed => {
                info!("Stop commanded: {}", command_id);
                self.transition_to(
                    SafetyState::Stopping, 
                    StateTransitionReason::UserCommand(format!("stop:{}", command_id))
                ).await?;
                self.pending_armed_timeout = None;
            }
            _ => {
                warn!("Cannot stop from state: {}", current_state.as_str());
            }
        }
        Ok(())
    }

    /// Complete stopping and return to idle
    pub async fn complete_stop(&mut self) -> Result<()> {
        let current_state = self.get_current_state().await;
        
        if current_state == SafetyState::Stopping {
            self.transition_to(SafetyState::Idle, StateTransitionReason::UserCommand("stop_complete".to_string())).await?;
        }
        Ok(())
    }

    /// Enter maintenance mode
    pub async fn enter_maintenance(&mut self) -> Result<()> {
        info!("Entering maintenance mode");
        self.transition_to(SafetyState::Maintenance, StateTransitionReason::MaintenanceMode).await?;
        Ok(())
    }

    /// Exit maintenance mode
    pub async fn exit_maintenance(&mut self) -> Result<()> {
        info!("Exiting maintenance mode");
        self.transition_to(SafetyState::Idle, StateTransitionReason::UserCommand("exit_maintenance".to_string())).await?;
        Ok(())
    }

    /// Record successful calibration with QC report
    pub async fn record_calibration_with_qc(&mut self, qc_report_json: String) -> Result<()> {
        self.last_calibration = Some(Utc::now());
        self.last_calibration_qc_json = Some(qc_report_json);
        info!("Calibration recorded at: {}", self.last_calibration.unwrap().to_rfc3339());
        
        // If we were locked due to calibration, transition to idle
        let current_state = self.get_current_state().await;
        if current_state == SafetyState::Locked {
            let interlocks = self.get_interlock_status().await;
            if interlocks.overall_safe {
                self.transition_to(SafetyState::Idle, StateTransitionReason::UserCommand("calibration_complete".to_string())).await?;
            }
        }
        
        Ok(())
    }
    
    /// Record successful calibration (legacy method without QC)
    pub async fn record_calibration(&mut self) -> Result<()> {
        self.record_calibration_with_qc("{}".to_string()).await
    }

    /// Check if calibration is valid (within 24 hours) - Public for API endpoints
    pub fn is_calibration_valid(&self) -> bool {
        if let Some(last_cal) = self.last_calibration {
            Utc::now() - last_cal < self.calibration_required_interval
        } else {
            false
        }
    }
    
    /// Public accessor: Get last calibration timestamp
    pub fn get_last_calibration(&self) -> Option<DateTime<Utc>> {
        self.last_calibration
    }
    
    /// Public accessor: Get last calibration QC report JSON
    pub fn get_last_calibration_qc_json(&self) -> Option<String> {
        self.last_calibration_qc_json.clone()
    }
    
    /// Public accessor: Get hours remaining until calibration expires
    pub fn get_time_until_expiry_hours(&self) -> Option<i64> {
        if let Some(last_cal) = self.last_calibration {
            let elapsed = Utc::now() - last_cal;
            let remaining = self.calibration_required_interval - elapsed;
            if remaining.num_seconds() > 0 {
                Some(remaining.num_hours())
            } else {
                Some(0) // Expired but show 0 instead of negative
            }
        } else {
            None // No calibration recorded
        }
    }

    /// Check if measurement requires physical enable (based on configuration)
    fn requires_physical_enable(&self) -> bool {
        // For now, always require physical enable for measurements
        // In production, this would check safety profile configuration
        true
    }

    /// Internal state transition with audit logging
    async fn transition_to(&mut self, new_state: SafetyState, reason: StateTransitionReason) -> Result<()> {
        let old_state = {
            self.current_state.read().await.clone()
        };

        if old_state == new_state {
            return Ok(()); // No change
        }

        {
            let mut state = self.current_state.write().await;
            *state = new_state.clone();
        }

        let interlocks = self.get_interlock_status().await;

        // Create audit log entry
        let state_change_log = CommandLog {
            id: uuid::Uuid::new_v4().to_string(),
            session_id: "safety_state_machine".to_string(),
            timestamp: Utc::now(),
            command_type: "state_transition".to_string(),
            command_data: serde_json::json!({
                "old_state": old_state.as_str(),
                "new_state": new_state.as_str(),
                "reason": reason.as_str(),
                "interlocks": {
                    "overall_safe": interlocks.overall_safe,
                    "key_switch": interlocks.key_switch,
                    "enable_button": interlocks.enable_button,
                    "emergency_stop": interlocks.emergency_stop,
                    "door_closed": interlocks.door_closed,
                    "beam_watchdog": interlocks.beam_watchdog,
                    "violation_reason": interlocks.violation_reason
                }
            }),
            user_context: Some("system".to_string()),
            device_state_before: serde_json::json!({"state": old_state.as_str()}),
            device_state_after: Some(serde_json::json!({"state": new_state.as_str()})),
            result: CommandResult::Success,
            execution_time_ms: 0,
        };

        if let Err(e) = self.audit_logger.log_command(state_change_log).await {
            error!("Failed to log state transition: {}", e);
        }

        info!("Safety state transition: {} -> {} ({})", 
              old_state.as_str(), new_state.as_str(), reason.as_str());

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    async fn create_test_state_machine() -> SafetyStateMachine {
        let temp_dir = TempDir::new().unwrap();
        let db_path = temp_dir.path().join("test_audit.db");
        let audit_logger = Arc::new(
            AuditLogger::new(db_path.to_str().unwrap(), "TEST-CERT".to_string())
                .await
                .unwrap(),
        );
        
        SafetyStateMachine::new(audit_logger)
    }

    #[tokio::test]
    async fn test_initial_state_locked() {
        let state_machine = create_test_state_machine().await;
        assert_eq!(state_machine.get_current_state().await, SafetyState::Locked);
    }

    #[tokio::test]
    async fn test_interlock_evaluation() {
        let mut status = InterlockStatus::new();
        status.key_switch = true;
        status.emergency_stop = true;
        status.door_closed = true;
        status.beam_watchdog = true;
        status.over_temperature = Some(true);
        status.evaluate_safety();
        
        assert!(status.overall_safe);
        assert!(status.violation_reason.is_none());
    }

    #[tokio::test]
    async fn test_calibration_requirement() {
        let mut state_machine = create_test_state_machine().await;
        
        // Without calibration, should block measurement
        let can_start = state_machine.start_measurement("test_cmd".to_string(), "test_user".to_string()).await.unwrap();
        assert!(!can_start);
        assert_eq!(state_machine.get_current_state().await, SafetyState::Locked);
    }
    
    #[tokio::test]
    async fn test_new_workflow_states() {
        let state_machine = create_test_state_machine().await;
        
        // Test Initialized state
        assert_eq!(SafetyState::Initialized.as_str(), "INITIALIZED");
        
        // Test WarmingUp state
        assert_eq!(SafetyState::WarmingUp.as_str(), "WARMING_UP");
        
        // Test Calibrated state
        assert_eq!(SafetyState::Calibrated.as_str(), "CALIBRATED");
        
        // Verify initial state
        assert_eq!(state_machine.get_current_state().await, SafetyState::Locked);
    }
    
    #[tokio::test]
    async fn test_state_transition_with_calibration() {
        let mut state_machine = create_test_state_machine().await;
        
        // Record calibration to allow measurements
        state_machine.record_calibration().await.unwrap();
        
        // Set interlocks to safe
        let mut interlocks = InterlockStatus::new();
        interlocks.key_switch = true;
        interlocks.emergency_stop = true;
        interlocks.door_closed = true;
        interlocks.beam_watchdog = true;
        interlocks.over_temperature = Some(true);
        interlocks.evaluate_safety();
        
        state_machine.update_interlocks(interlocks).await.unwrap();
        
        // Initialize to idle state first
        state_machine.transition_to(
            SafetyState::Idle,
            StateTransitionReason::SystemStartup
        ).await.unwrap();
        
        // Now should be able to start measurement
        let can_start = state_machine.start_measurement(
            "test_cmd".to_string(),
            "test_user".to_string()
        ).await.unwrap();
        
        assert!(can_start);
        assert_eq!(state_machine.get_current_state().await, SafetyState::PendingArmed);
    }
}
