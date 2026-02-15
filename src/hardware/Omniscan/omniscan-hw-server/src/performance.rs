//! Performance monitoring and metrics for medical device operations
//! 
//! Implements requirements from USER_EXPECTATIONS.md Section 11:
//! - Measurement initiation: < 2 seconds
//! - Emergency stop response: < 100 milliseconds
//! - State transition display: < 1 second
//! - Data commit to storage: < 5 seconds
//! - Calibration routine: < 10 minutes
//! - Historical data search: < 5 seconds

use std::time::Instant;
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use serde::{Deserialize, Serialize};
use chrono::{DateTime, Utc};

/// Performance benchmark thresholds per USER_EXPECTATIONS.md Section 11
pub struct PerformanceThresholds {
    pub measurement_initiation_ms: u64,      // < 2000 ms
    pub emergency_stop_response_ms: u64,     // < 100 ms
    pub state_transition_display_ms: u64,    // < 1000 ms
    pub data_commit_storage_ms: u64,         // < 5000 ms
    pub calibration_routine_ms: u64,         // < 600000 ms (10 minutes)
    pub historical_data_search_ms: u64,      // < 5000 ms
}

impl Default for PerformanceThresholds {
    fn default() -> Self {
        Self {
            measurement_initiation_ms: 2000,
            emergency_stop_response_ms: 100,
            state_transition_display_ms: 1000,
            data_commit_storage_ms: 5000,
            calibration_routine_ms: 600000,
            historical_data_search_ms: 5000,
        }
    }
}

/// Operation types for performance tracking
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum OperationType {
    MeasurementInitiation,
    EmergencyStopResponse,
    StateTransitionDisplay,
    DataCommitStorage,
    CalibrationRoutine,
    HistoricalDataSearch,
    // Additional operations
    DetectorAcquisition,
    MotionControl,
    InterlockCheck,
}

impl OperationType {
    /// Get the performance threshold for this operation type
    pub fn threshold_ms(&self, thresholds: &PerformanceThresholds) -> u64 {
        match self {
            OperationType::MeasurementInitiation => thresholds.measurement_initiation_ms,
            OperationType::EmergencyStopResponse => thresholds.emergency_stop_response_ms,
            OperationType::StateTransitionDisplay => thresholds.state_transition_display_ms,
            OperationType::DataCommitStorage => thresholds.data_commit_storage_ms,
            OperationType::CalibrationRoutine => thresholds.calibration_routine_ms,
            OperationType::HistoricalDataSearch => thresholds.historical_data_search_ms,
            _ => 5000, // Default 5 second threshold for other operations
        }
    }
    
    /// Get human-readable name
    pub fn name(&self) -> &'static str {
        match self {
            OperationType::MeasurementInitiation => "Measurement Initiation",
            OperationType::EmergencyStopResponse => "Emergency Stop Response",
            OperationType::StateTransitionDisplay => "State Transition Display",
            OperationType::DataCommitStorage => "Data Commit Storage",
            OperationType::CalibrationRoutine => "Calibration Routine",
            OperationType::HistoricalDataSearch => "Historical Data Search",
            OperationType::DetectorAcquisition => "Detector Acquisition",
            OperationType::MotionControl => "Motion Control",
            OperationType::InterlockCheck => "Interlock Check",
        }
    }
}

/// Performance measurement record
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceMeasurement {
    pub operation_type: OperationType,
    pub start_time: DateTime<Utc>,
    pub duration_ms: u64,
    pub threshold_ms: u64,
    pub exceeded_threshold: bool,
    pub metadata: HashMap<String, String>,
}

/// Performance statistics for an operation type
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceStats {
    pub operation_type: OperationType,
    pub total_operations: usize,
    pub threshold_violations: usize,
    pub min_duration_ms: u64,
    pub max_duration_ms: u64,
    pub avg_duration_ms: u64,
    pub p50_duration_ms: u64,  // Median
    pub p95_duration_ms: u64,  // 95th percentile
    pub p99_duration_ms: u64,  // 99th percentile
}

/// Performance monitor for tracking operation timings
pub struct PerformanceMonitor {
    thresholds: PerformanceThresholds,
    measurements: Arc<RwLock<Vec<PerformanceMeasurement>>>,
    max_measurements: usize,
}

impl PerformanceMonitor {
    pub fn new() -> Self {
        Self {
            thresholds: PerformanceThresholds::default(),
            measurements: Arc::new(RwLock::new(Vec::new())),
            max_measurements: 10000, // Keep last 10k measurements
        }
    }
    
    /// Start timing an operation
    pub fn start_operation(&self, operation_type: OperationType) -> OperationTimer {
        OperationTimer {
            operation_type,
            start_instant: Instant::now(),
            start_time: Utc::now(),
            threshold_ms: operation_type.threshold_ms(&self.thresholds),
            metadata: HashMap::new(),
        }
    }
    
    /// Record completed operation
    pub async fn record_operation(&self, timer: OperationTimer, metadata: Option<HashMap<String, String>>) {
        let duration_ms = timer.elapsed_ms();
        let exceeded = duration_ms > timer.threshold_ms;
        
        let measurement = PerformanceMeasurement {
            operation_type: timer.operation_type,
            start_time: timer.start_time,
            duration_ms,
            threshold_ms: timer.threshold_ms,
            exceeded_threshold: exceeded,
            metadata: metadata.unwrap_or_else(|| timer.metadata.clone()),
        };
        
        // Log warning if threshold exceeded
        if exceeded {
            tracing::warn!(
                "Performance threshold exceeded: {} took {}ms (threshold: {}ms)",
                timer.operation_type.name(),
                duration_ms,
                timer.threshold_ms
            );
        }
        
        let mut measurements = self.measurements.write().await;
        measurements.push(measurement);
        
        // Keep only recent measurements
        if measurements.len() > self.max_measurements {
            let excess = measurements.len() - self.max_measurements;
            measurements.drain(0..excess);
        }
    }
    
    /// Get statistics for an operation type
    pub async fn get_stats(&self, operation_type: OperationType) -> Option<PerformanceStats> {
        let measurements = self.measurements.read().await;
        
        let ops: Vec<_> = measurements.iter()
            .filter(|m| m.operation_type == operation_type)
            .collect();
        
        if ops.is_empty() {
            return None;
        }
        
        let mut durations: Vec<u64> = ops.iter().map(|m| m.duration_ms).collect();
        durations.sort_unstable();
        
        let total = durations.len();
        let violations = ops.iter().filter(|m| m.exceeded_threshold).count();
        let sum: u64 = durations.iter().sum();
        
        Some(PerformanceStats {
            operation_type,
            total_operations: total,
            threshold_violations: violations,
            min_duration_ms: durations[0],
            max_duration_ms: durations[total - 1],
            avg_duration_ms: sum / total as u64,
            p50_duration_ms: durations[total / 2],
            p95_duration_ms: durations[(total as f64 * 0.95) as usize],
            p99_duration_ms: durations[(total as f64 * 0.99) as usize],
        })
    }
    
    /// Get all statistics
    pub async fn get_all_stats(&self) -> Vec<PerformanceStats> {
        let mut stats = Vec::new();
        
        for op_type in [
            OperationType::MeasurementInitiation,
            OperationType::EmergencyStopResponse,
            OperationType::StateTransitionDisplay,
            OperationType::DataCommitStorage,
            OperationType::CalibrationRoutine,
            OperationType::HistoricalDataSearch,
        ] {
            if let Some(stat) = self.get_stats(op_type).await {
                stats.push(stat);
            }
        }
        
        stats
    }
    
    /// Check if recent performance is acceptable
    pub async fn check_health(&self) -> PerformanceHealth {
        let measurements = self.measurements.read().await;
        
        // Check last 100 operations
        let recent: Vec<_> = measurements.iter()
            .rev()
            .take(100)
            .collect();
        
        if recent.is_empty() {
            return PerformanceHealth {
                status: HealthStatus::Unknown,
                details: "No performance data available".to_string(),
            };
        }
        
        let violations = recent.iter()
            .filter(|m| m.exceeded_threshold)
            .count();
        
        let violation_rate = violations as f64 / recent.len() as f64;
        
        let (status, details) = if violation_rate > 0.2 {
            (HealthStatus::Critical, format!(
                "High performance degradation: {:.1}% of operations exceeding thresholds",
                violation_rate * 100.0
            ))
        } else if violation_rate > 0.05 {
            (HealthStatus::Warning, format!(
                "Some performance issues: {:.1}% of operations exceeding thresholds",
                violation_rate * 100.0
            ))
        } else {
            (HealthStatus::Healthy, format!(
                "Performance healthy: {:.1}% threshold violations",
                violation_rate * 100.0
            ))
        };
        
        PerformanceHealth { status, details }
    }
}

impl Default for PerformanceMonitor {
    fn default() -> Self {
        Self::new()
    }
}

/// Timer for measuring operation duration
pub struct OperationTimer {
    operation_type: OperationType,
    start_instant: Instant,
    start_time: DateTime<Utc>,
    threshold_ms: u64,
    metadata: HashMap<String, String>,
}

impl OperationTimer {
    /// Add metadata to this operation
    pub fn add_metadata(&mut self, key: String, value: String) {
        self.metadata.insert(key, value);
    }
    
    /// Get elapsed time in milliseconds
    pub fn elapsed_ms(&self) -> u64 {
        self.start_instant.elapsed().as_millis() as u64
    }
    
    /// Check if threshold has been exceeded
    pub fn is_exceeded(&self) -> bool {
        self.elapsed_ms() > self.threshold_ms
    }
}

/// Performance health status
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceHealth {
    pub status: HealthStatus,
    pub details: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum HealthStatus {
    Healthy,
    Warning,
    Critical,
    Unknown,
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio::time::Duration;
    
    #[tokio::test]
    async fn test_performance_monitoring() {
        let monitor = PerformanceMonitor::new();
        
        // Simulate fast operation
        let timer = monitor.start_operation(OperationType::MeasurementInitiation);
        tokio::time::sleep(Duration::from_millis(100)).await;
        monitor.record_operation(timer, None).await;
        
        // Simulate slow operation
        let timer = monitor.start_operation(OperationType::MeasurementInitiation);
        tokio::time::sleep(Duration::from_millis(3000)).await;
        monitor.record_operation(timer, None).await;
        
        let stats = monitor.get_stats(OperationType::MeasurementInitiation).await.unwrap();
        assert_eq!(stats.total_operations, 2);
        assert_eq!(stats.threshold_violations, 1); // One exceeded 2s threshold
    }
    
    #[test]
    fn test_operation_threshold() {
        let thresholds = PerformanceThresholds::default();
        assert_eq!(
            OperationType::EmergencyStopResponse.threshold_ms(&thresholds),
            100
        );
        assert_eq!(
            OperationType::MeasurementInitiation.threshold_ms(&thresholds),
            2000
        );
    }
}
