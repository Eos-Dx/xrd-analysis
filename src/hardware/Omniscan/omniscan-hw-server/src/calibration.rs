//! Calibration management for medical device operations
//! 
//! Implements requirements from USER_EXPECTATIONS.md Section 4:
//! - Daily calibration enforcement (24-hour rule)
//! - Block diagnostic measurements without valid calibration
//! - Guided workflow with step-by-step instructions
//! - Automatic validation against acceptance criteria
//! - Clear pass/fail indication with corrective guidance
//! - Transition to LOCKED state when calibration expires
//! - Display detailed calibration parameters
//! - Graphical trending of calibration results
//! - Historical data for failure pattern analysis

use std::sync::Arc;
use tokio::sync::RwLock;
use chrono::{DateTime, Utc, Duration};
use serde::{Deserialize, Serialize};
use uuid::Uuid;
use anyhow::Result;

/// Calibration status
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CalibrationStatus {
    NotCalibrated,
    Valid,
    Expired,
    Failed,
    InProgress,
}

impl CalibrationStatus {
    pub fn as_str(&self) -> &'static str {
        match self {
            CalibrationStatus::NotCalibrated => "NOT_CALIBRATED",
            CalibrationStatus::Valid => "VALID",
            CalibrationStatus::Expired => "EXPIRED",
            CalibrationStatus::Failed => "FAILED",
            CalibrationStatus::InProgress => "IN_PROGRESS",
        }
    }
    
    /// Plain language description for operators (Section 5)
    pub fn user_message(&self) -> &'static str {
        match self {
            CalibrationStatus::NotCalibrated => "Daily calibration has not been performed",
            CalibrationStatus::Valid => "Calibration is valid and current",
            CalibrationStatus::Expired => "Calibration has expired. Run daily calibration before measurements.",
            CalibrationStatus::Failed => "Calibration failed validation. Contact service engineer.",
            CalibrationStatus::InProgress => "Calibration is currently in progress",
        }
    }
}

/// Calibration acceptance criteria
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AcceptanceCriteria {
    /// Beam center X position (mm) - acceptable range
    pub beam_center_x_min: f64,
    pub beam_center_x_max: f64,
    
    /// Beam center Y position (mm) - acceptable range
    pub beam_center_y_min: f64,
    pub beam_center_y_max: f64,
    
    /// Sample-to-detector distance (mm) - acceptable range
    pub distance_min: f64,
    pub distance_max: f64,
    
    /// Peak intensity (counts) - minimum threshold
    pub min_peak_intensity: u32,
    
    /// Signal-to-noise ratio - minimum threshold
    pub min_snr: f64,
    
    /// Peak position error (mm) - maximum acceptable
    pub max_peak_position_error: f64,
}

impl Default for AcceptanceCriteria {
    fn default() -> Self {
        Self {
            beam_center_x_min: -2.0,
            beam_center_x_max: 2.0,
            beam_center_y_min: -2.0,
            beam_center_y_max: 2.0,
            distance_min: 98.0,
            distance_max: 102.0,
            min_peak_intensity: 1000,
            min_snr: 10.0,
            max_peak_position_error: 0.5,
        }
    }
}

/// Calibration measurement parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CalibrationMeasurement {
    pub beam_center_x: f64,
    pub beam_center_y: f64,
    pub sample_distance: f64,
    pub peak_intensity: u32,
    pub snr: f64,
    pub peak_position_error: f64,
    /// Standard used for calibration (e.g., "LaB6", "Si")
    pub calibrant_standard: String,
}

/// Validation result for calibration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationResult {
    pub passed: bool,
    pub failures: Vec<String>,
    pub warnings: Vec<String>,
}

impl ValidationResult {
    /// Get corrective action guidance for operator
    pub fn corrective_actions(&self) -> Vec<String> {
        if self.passed {
            return vec!["Calibration passed. System ready for measurements.".to_string()];
        }
        
        let mut actions = vec![
            "Calibration failed validation. Review the following:".to_string(),
        ];
        
        for (i, failure) in self.failures.iter().enumerate() {
            actions.push(format!("{}. {}", i + 1, failure));
        }
        
        actions.push("If issues persist after retry, contact service engineer.".to_string());
        
        actions
    }
}

/// Complete calibration record
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CalibrationRecord {
    pub id: String,
    pub timestamp: DateTime<Utc>,
    pub operator: String,
    pub measurement: CalibrationMeasurement,
    pub validation: ValidationResult,
    pub status: CalibrationStatus,
    pub notes: Option<String>,
    pub device_serial: String,
}

/// Calibration manager
pub struct CalibrationManager {
    current_calibration: Arc<RwLock<Option<CalibrationRecord>>>,
    calibration_history: Arc<RwLock<Vec<CalibrationRecord>>>,
    acceptance_criteria: Arc<RwLock<AcceptanceCriteria>>,
    validity_duration: Duration,
    demo_mode: bool,
}

impl CalibrationManager {
    pub fn new() -> Self {
        Self {
            current_calibration: Arc::new(RwLock::new(None)),
            calibration_history: Arc::new(RwLock::new(Vec::new())),
            acceptance_criteria: Arc::new(RwLock::new(AcceptanceCriteria::default())),
            validity_duration: Duration::hours(24), // 24-hour rule
            demo_mode: false,
        }
    }
    
    /// Create a new CalibrationManager with demo mode support
    pub fn with_demo_mode(demo_mode: bool) -> Self {
        Self {
            current_calibration: Arc::new(RwLock::new(None)),
            calibration_history: Arc::new(RwLock::new(Vec::new())),
            acceptance_criteria: Arc::new(RwLock::new(AcceptanceCriteria::default())),
            validity_duration: Duration::hours(24), // 24-hour rule
            demo_mode,
        }
    }
    
    /// Check if current calibration is valid
    pub async fn is_valid(&self) -> bool {
        let cal = self.current_calibration.read().await;
        
        if let Some(record) = cal.as_ref() {
            if record.status != CalibrationStatus::Valid {
                return false;
            }
            
            let elapsed = Utc::now() - record.timestamp;
            elapsed < self.validity_duration
        } else {
            false
        }
    }
    
    /// Get current calibration status
    pub async fn get_status(&self) -> CalibrationStatus {
        let cal = self.current_calibration.read().await;
        
        match cal.as_ref() {
            None => CalibrationStatus::NotCalibrated,
            Some(record) => {
                if record.status == CalibrationStatus::Valid {
                    let elapsed = Utc::now() - record.timestamp;
                    if elapsed >= self.validity_duration {
                        CalibrationStatus::Expired
                    } else {
                        CalibrationStatus::Valid
                    }
                } else {
                    record.status
                }
            }
        }
    }
    
    /// Get time until calibration expires
    pub async fn time_until_expiry(&self) -> Option<Duration> {
        let cal = self.current_calibration.read().await;
        
        if let Some(record) = cal.as_ref() {
            if record.status == CalibrationStatus::Valid {
                let elapsed = Utc::now() - record.timestamp;
                let remaining = self.validity_duration - elapsed;
                return if remaining > Duration::zero() {
                    Some(remaining)
                } else {
                    Some(Duration::zero())
                };
            }
        }
        
        None
    }
    
    /// Start calibration procedure
    /// TODO: Integrate with actual detector and motion control
    pub async fn start_calibration(&self, operator: String) -> Result<String> {
        tracing::info!("Starting calibration procedure for operator: {}", operator);
        
        // TODO: Implement actual calibration workflow:
        // 1. Verify calibrant standard is loaded
        // 2. Move to calibration position
        // 3. Acquire calibration image
        // 4. Analyze peaks and determine parameters
        // 5. Validate against acceptance criteria
        
        // For now, return calibration ID
        let calibration_id = Uuid::new_v4().to_string();
        
        Ok(calibration_id)
    }
    
    /// Record calibration measurement and validate
    pub async fn record_calibration(
        &self,
        operator: String,
        measurement: CalibrationMeasurement,
        device_serial: String,
    ) -> Result<CalibrationRecord> {
        tracing::info!("Recording calibration measurement");
        
        // Validate measurement
        let validation = self.validate_measurement(&measurement).await;
        
        let status = if validation.passed {
            CalibrationStatus::Valid
        } else {
            CalibrationStatus::Failed
        };
        
        let record = CalibrationRecord {
            id: Uuid::new_v4().to_string(),
            timestamp: Utc::now(),
            operator,
            measurement,
            validation,
            status,
            notes: None,
            device_serial,
        };
        
        // Update current calibration if passed
        if status == CalibrationStatus::Valid {
            let mut current = self.current_calibration.write().await;
            *current = Some(record.clone());
            tracing::info!("Calibration passed validation - device ready for measurements");
        } else {
            tracing::warn!("Calibration failed validation");
        }
        
        // Add to history
        let mut history = self.calibration_history.write().await;
        history.push(record.clone());
        
        Ok(record)
    }
    
    /// Validate calibration measurement against acceptance criteria
    async fn validate_measurement(&self, measurement: &CalibrationMeasurement) -> ValidationResult {
        let criteria = self.acceptance_criteria.read().await;
        let mut failures = Vec::new();
        let mut warnings = Vec::new();
        
        // Check beam center X
        if measurement.beam_center_x < criteria.beam_center_x_min 
           || measurement.beam_center_x > criteria.beam_center_x_max {
            failures.push(format!(
                "Beam center X ({:.2} mm) outside acceptable range ({:.2} to {:.2} mm)",
                measurement.beam_center_x,
                criteria.beam_center_x_min,
                criteria.beam_center_x_max
            ));
        }
        
        // Check beam center Y
        if measurement.beam_center_y < criteria.beam_center_y_min 
           || measurement.beam_center_y > criteria.beam_center_y_max {
            failures.push(format!(
                "Beam center Y ({:.2} mm) outside acceptable range ({:.2} to {:.2} mm)",
                measurement.beam_center_y,
                criteria.beam_center_y_min,
                criteria.beam_center_y_max
            ));
        }
        
        // Check sample distance (auto-pass in demo mode)
        if !self.demo_mode {
            if measurement.sample_distance < criteria.distance_min 
               || measurement.sample_distance > criteria.distance_max {
                failures.push(format!(
                    "Sample distance ({:.2} mm) outside acceptable range ({:.2} to {:.2} mm)",
                    measurement.sample_distance,
                    criteria.distance_min,
                    criteria.distance_max
                ));
            }
        } else {
            tracing::debug!("Demo mode: Auto-passing distance check for distance {:.2} mm", measurement.sample_distance);
        }
        
        // Check peak intensity
        if measurement.peak_intensity < criteria.min_peak_intensity {
            failures.push(format!(
                "Peak intensity ({} counts) below minimum ({})",
                measurement.peak_intensity,
                criteria.min_peak_intensity
            ));
        }
        
        // Check SNR
        if measurement.snr < criteria.min_snr {
            failures.push(format!(
                "Signal-to-noise ratio ({:.1}) below minimum ({:.1})",
                measurement.snr,
                criteria.min_snr
            ));
        }
        
        // Check peak position error
        if measurement.peak_position_error > criteria.max_peak_position_error {
            warnings.push(format!(
                "Peak position error ({:.3} mm) higher than ideal ({:.3} mm)",
                measurement.peak_position_error,
                criteria.max_peak_position_error
            ));
        }
        
        ValidationResult {
            passed: failures.is_empty(),
            failures,
            warnings,
        }
    }
    
    /// Get calibration history
    pub async fn get_history(&self, limit: usize) -> Vec<CalibrationRecord> {
        let history = self.calibration_history.read().await;
        history.iter()
            .rev()
            .take(limit)
            .cloned()
            .collect()
    }
    
    /// Get calibration trend data for analysis
    pub async fn get_trend_data(&self, days: i64) -> Vec<CalibrationTrendPoint> {
        let history = self.calibration_history.read().await;
        let cutoff = Utc::now() - Duration::days(days);
        
        history.iter()
            .filter(|r| r.timestamp > cutoff)
            .map(|r| CalibrationTrendPoint {
                timestamp: r.timestamp,
                beam_center_x: r.measurement.beam_center_x,
                beam_center_y: r.measurement.beam_center_y,
                sample_distance: r.measurement.sample_distance,
                peak_intensity: r.measurement.peak_intensity,
                snr: r.measurement.snr,
                passed: r.validation.passed,
            })
            .collect()
    }
    
    /// Update acceptance criteria (admin only)
    pub async fn update_criteria(&self, criteria: AcceptanceCriteria) -> Result<()> {
        tracing::info!("Updating calibration acceptance criteria");
        let mut current = self.acceptance_criteria.write().await;
        *current = criteria;
        Ok(())
    }
}

impl Default for CalibrationManager {
    fn default() -> Self {
        Self::new()
    }
}

/// Data point for calibration trending
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CalibrationTrendPoint {
    pub timestamp: DateTime<Utc>,
    pub beam_center_x: f64,
    pub beam_center_y: f64,
    pub sample_distance: f64,
    pub peak_intensity: u32,
    pub snr: f64,
    pub passed: bool,
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[tokio::test]
    async fn test_calibration_validation() {
        let manager = CalibrationManager::new();
        
        // Valid measurement
        let valid_measurement = CalibrationMeasurement {
            beam_center_x: 0.5,
            beam_center_y: -0.3,
            sample_distance: 100.0,
            peak_intensity: 5000,
            snr: 25.0,
            peak_position_error: 0.1,
            calibrant_standard: "LaB6".to_string(),
        };
        
        let record = manager.record_calibration(
            "test_operator".to_string(),
            valid_measurement,
            "DEVICE-001".to_string()
        ).await.unwrap();
        
        assert_eq!(record.status, CalibrationStatus::Valid);
        assert!(record.validation.passed);
        assert!(manager.is_valid().await);
    }
    
    #[tokio::test]
    async fn test_calibration_distance_validation_passes() {
        let manager = CalibrationManager::new();
        
        // Test distance within acceptable range (98.0 to 102.0 mm)
        let measurement = CalibrationMeasurement {
            beam_center_x: 0.0,
            beam_center_y: 0.0,
            sample_distance: 100.0,  // Within range
            peak_intensity: 5000,
            snr: 25.0,
            peak_position_error: 0.1,
            calibrant_standard: "LaB6".to_string(),
        };
        
        let record = manager.record_calibration(
            "test_operator".to_string(),
            measurement,
            "DEVICE-001".to_string()
        ).await.unwrap();
        
        // Distance test should pass
        assert!(record.validation.passed, "Distance validation should pass for distance within range");
        assert!(record.validation.failures.is_empty(), "No failures should be reported");
    }
    
    #[tokio::test]
    async fn test_calibration_distance_validation_fails_too_close() {
        let manager = CalibrationManager::new();
        
        // Test distance below acceptable range
        let measurement = CalibrationMeasurement {
            beam_center_x: 0.0,
            beam_center_y: 0.0,
            sample_distance: 95.0,  // Below minimum (98.0)
            peak_intensity: 5000,
            snr: 25.0,
            peak_position_error: 0.1,
            calibrant_standard: "LaB6".to_string(),
        };
        
        let record = manager.record_calibration(
            "test_operator".to_string(),
            measurement,
            "DEVICE-001".to_string()
        ).await.unwrap();
        
        // Distance test should fail
        assert!(!record.validation.passed, "Distance validation should fail for distance below minimum");
        assert!(record.validation.failures.iter().any(|f| f.contains("95.0")), "Failure should mention the distance");
    }
    
    #[tokio::test]
    async fn test_calibration_distance_validation_fails_too_far() {
        let manager = CalibrationManager::new();
        
        // Test distance above acceptable range
        let measurement = CalibrationMeasurement {
            beam_center_x: 0.0,
            beam_center_y: 0.0,
            sample_distance: 105.0,  // Above maximum (102.0)
            peak_intensity: 5000,
            snr: 25.0,
            peak_position_error: 0.1,
            calibrant_standard: "LaB6".to_string(),
        };
        
        let record = manager.record_calibration(
            "test_operator".to_string(),
            measurement,
            "DEVICE-001".to_string()
        ).await.unwrap();
        
        // Distance test should fail
        assert!(!record.validation.passed, "Distance validation should fail for distance above maximum");
        assert!(record.validation.failures.iter().any(|f| f.contains("105.0")), "Failure should mention the distance");
    }
    
    #[tokio::test]
    async fn test_calibration_distance_auto_pass_in_demo_mode() {
        let manager = CalibrationManager::with_demo_mode(true);
        
        // Test distance way out of range but in demo mode
        let measurement = CalibrationMeasurement {
            beam_center_x: 0.0,
            beam_center_y: 0.0,
            sample_distance: 500.0,  // Way above maximum (102.0)
            peak_intensity: 5000,
            snr: 25.0,
            peak_position_error: 0.1,
            calibrant_standard: "LaB6".to_string(),
        };
        
        let record = manager.record_calibration(
            "test_operator".to_string(),
            measurement,
            "DEVICE-001".to_string()
        ).await.unwrap();
        
        // Distance test should PASS in demo mode even though value is out of range
        assert!(record.validation.passed, "Distance validation should auto-pass in demo mode");
        assert!(!record.validation.failures.iter().any(|f| f.contains("500.0")), "No distance failure should be reported in demo mode");
    }
    
    #[tokio::test]
    async fn test_calibration_expiry() {
        let mut manager = CalibrationManager::new();
        // Set short validity for testing
        manager.validity_duration = Duration::seconds(1);
        
        let measurement = CalibrationMeasurement {
            beam_center_x: 0.0,
            beam_center_y: 0.0,
            sample_distance: 100.0,
            peak_intensity: 5000,
            snr: 25.0,
            peak_position_error: 0.1,
            calibrant_standard: "LaB6".to_string(),
        };
        
        manager.record_calibration(
            "operator".to_string(),
            measurement,
            "DEVICE-001".to_string()
        ).await.unwrap();
        
        assert!(manager.is_valid().await);
        
        // Wait for expiry
        tokio::time::sleep(tokio::time::Duration::from_secs(2)).await;
        
        assert!(!manager.is_valid().await);
        assert_eq!(manager.get_status().await, CalibrationStatus::Expired);
    }
}
