use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::time::Duration;
use thiserror::Error;

pub mod demo;
pub mod test;
pub mod bruker_axscom;
// Future detector implementations:
// pub mod bruker_bis;
// pub mod advocam;

// Re-export all detector implementations
pub use demo::DemoDetector;
pub use test::TestDetector;
pub use bruker_axscom::BrukerAxscomDetector;

#[derive(Debug, Error)]
pub enum DetectorError {
    #[error("Detector is not powered on")]
    NotPowered,
    #[error("Detector is already running an exposure")]
    ExposureInProgress,
    #[error("Invalid exposure time: {0}ms")]
    InvalidExposureTime(u32),
    #[error("Hardware communication error: {0}")]
    HardwareError(String),
    #[error("Calibration error: {0}")]
    CalibrationError(String),
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum DetectorStatus {
    Off,
    Init,        // Powered but initializing (not yet ready)
    Idle,
    Exposing,
    Reading,
    Error(String),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DetectorHealth {
    pub powered: bool,
    pub temperature: f32, // Celsius
    pub voltage: f32,     // Volts
    pub status: DetectorStatus,
    pub last_exposure_time: Option<u32>, // milliseconds
    pub total_exposures: u64,
    pub uptime: Duration,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExposureResult {
    pub exposure_time_ms: u32,
    pub timestamp: chrono::DateTime<chrono::Utc>,
    pub data_size: usize,
    pub data_path: Option<String>,
    pub detector_temp: f32,
}

/// Trait for detector devices
#[async_trait]
pub trait DetectorDevice: Send + Sync {
    /// Power on the detector
    async fn power_on(&self) -> Result<(), DetectorError>;
    
    /// Power off the detector
    async fn power_off(&self) -> Result<(), DetectorError>;
    
    /// Check if detector is powered
    async fn is_powered(&self) -> bool;
    
    /// Get current detector status
    async fn get_status(&self) -> DetectorStatus;
    
    /// Get detector health information
    async fn get_health(&self) -> DetectorHealth;
    
    /// Start an exposure with the given time in milliseconds
    async fn start_exposure(&self, exposure_time_ms: u32) -> Result<(), DetectorError>;
    
    /// Stop current exposure (if any)
    async fn stop_exposure(&self) -> Result<(), DetectorError>;
    
    /// Get the result of the last completed exposure
    async fn get_last_result(&self) -> Option<ExposureResult>;
    
    /// Calibrate the detector
    async fn calibrate(&self) -> Result<(), DetectorError>;
}