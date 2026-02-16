//! OMNIScan Medical Device Hardware Server Library
//! 
//! FDA/IEC 62304 Class B compliant medical device software for X-ray diffraction analysis.
//! This crate provides safety-critical device control, audit logging, and gRPC communication.
//!
//! Aligned with USER_EXPECTATIONS.md requirements for clinical medical device software.

pub mod audit;
pub mod auth;
pub mod calibration;
pub mod calibration_qc;
pub mod certificates;
pub mod commands;
pub mod config;
pub mod devices;
pub mod encryption;
pub mod errors;
pub mod grpc;
pub mod logging;
pub mod performance;
pub mod safety;
pub mod warmup;

// Re-export main types for medical device operations
pub use audit::{AuditLogger, CommandLog, CommandResult};
pub use config::ServerConfig;
pub use safety::{SafetyStateMachine, SafetyState, InterlockStatus};

// Re-export device traits and implementations
pub use devices::{
    DetectorDevice, MotionControlDevice, GpioDevice,
    detectors::DemoDetector,
    motions::{DemoMotion, XYDemoMotion},
    gpio::DemoGpio,
};
