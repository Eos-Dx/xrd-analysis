//! Device abstractions and implementations for the Omniscan hardware server
//! 
//! This module provides trait definitions for various hardware components
//! and their implementations (both DEMO and test versions).

pub mod detectors;
pub mod motions;
pub mod gpio;
pub mod pdu;

// Re-export main traits for convenience
pub use detectors::{DetectorDevice, DetectorStatus};
pub use motions::MotionControlDevice;
pub use gpio::{GpioDevice, InterlockStatus};

// Re-export DEMO implementations
pub use gpio::DemoGpio;
pub use pdu::DemoPdu;
