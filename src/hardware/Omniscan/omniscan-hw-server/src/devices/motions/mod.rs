use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::time::Duration;
use thiserror::Error;

pub mod demo;
pub mod xy_demo;
pub mod test;
// Future motion implementations:
// pub mod thorlabs;
// pub mod newport;

// Re-export all motion implementations
pub use demo::DemoMotion;
pub use xy_demo::XYDemoMotion;
pub use test::TestMotion;

#[derive(Debug, Error)]
pub enum MotionError {
    #[error("Motion controller is not powered on")]
    NotPowered,
    #[error("Motion is already in progress")]
    MotionInProgress,
    #[error("Invalid position: {0}")]
    InvalidPosition(f64),
    #[error("Motion limit reached: {0}")]
    LimitReached(String),
    #[error("Hardware communication error: {0}")]
    HardwareError(String),
    #[error("Homing required")]
    HomingRequired,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum MotionStatus {
    Off,
    Init,        // Powered but initializing (not yet ready)
    Idle,
    Moving,
    Homing,
    Error(String),
    LimitHit,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MotionHealth {
    pub powered: bool,
    pub status: MotionStatus,
    pub position: Option<f64>, // Current position in mm
    pub target_position: Option<f64>,
    pub is_homed: bool,
    pub total_moves: u64,
    pub uptime: Duration,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MotionLimits {
    pub min_position: f64, // mm
    pub max_position: f64, // mm
    pub max_velocity: f64, // mm/s
    pub max_acceleration: f64, // mm/s²
}

/// Trait for motion control devices
#[async_trait]
pub trait MotionControlDevice: Send + Sync {
    /// Power on the motion controller
    async fn power_on(&self) -> Result<(), MotionError>;
    
    /// Power off the motion controller
    async fn power_off(&self) -> Result<(), MotionError>;
    
    /// Check if motion controller is powered
    async fn is_powered(&self) -> bool;
    
    /// Get current motion status
    async fn get_status(&self) -> MotionStatus;
    
    /// Get motion controller health information
    async fn get_health(&self) -> MotionHealth;
    
    /// Home the motion controller (find reference position)
    async fn home(&self) -> Result<(), MotionError>;
    
    /// Move to absolute position in mm
    async fn move_to(&self, position_mm: f64) -> Result<(), MotionError>;
    
    /// Move relative to current position in mm
    async fn move_relative(&self, distance_mm: f64) -> Result<(), MotionError>;
    
    /// Stop current motion
    async fn stop_motion(&self) -> Result<(), MotionError>;
    
    /// Get current position in mm
    async fn get_position(&self) -> Option<f64>;
    
    /// Get motion limits
    async fn get_limits(&self) -> MotionLimits;
    
    /// Set motion velocity in mm/s
    async fn set_velocity(&self, velocity_mm_s: f64) -> Result<(), MotionError>;
}