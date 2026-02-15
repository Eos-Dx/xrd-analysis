use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::time::Duration;
use thiserror::Error;

pub mod demo;
pub mod test;
#[cfg(feature = "gui")]
pub mod gui;
// Future GPIO implementations:
// pub mod pcie_card;
// pub mod usb_relay;

// Re-export all GPIO implementations
pub use demo::DemoGpio;
pub use test::TestGpio;
#[cfg(feature = "gui")]
pub use gui::GpioControlGui;

#[derive(Debug, Error)]
pub enum GpioError {
    #[error("GPIO controller is not powered on")]
    NotPowered,
    #[error("Invalid pin number: {0}")]
    InvalidPin(u8),
    #[error("Pin {0} is not configured for this operation")]
    PinNotConfigured(u8),
    #[error("Hardware communication error: {0}")]
    HardwareError(String),
    #[error("Interlock violation: {0}")]
    InterlockViolation(String),
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum GpioPinMode {
    Input,
    Output,
    InputPullUp,
    InputPullDown,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub enum GpioPinState {
    High,
    Low,
}

/// LED colors for status indicators
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub enum LedColor {
    Red,
    Orange,
    Green,
    Off,
}

/// LED identifiers
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub enum Led {
    MainStatus,
    RadiationWarning,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpioPin {
    pub pin: u8,
    pub mode: GpioPinMode,
    pub state: GpioPinState,
    pub name: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpioHealth {
    pub powered: bool,
    pub total_pins: u8,
    pub configured_pins: usize,
    pub interlocks_status: InterlockStatus,
    pub uptime: Duration,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InterlockStatus {
    pub emergency_stop: bool,
    pub door_closed: bool,
    pub radiation_safe: bool,
    pub cooling_ok: bool,
    pub power_ok: bool,
    pub overall_safe: bool,
}

/// Trait for GPIO devices
#[async_trait]
pub trait GpioDevice: Send + Sync {
    /// Power on the GPIO controller
    async fn power_on(&self) -> Result<(), GpioError>;
    
    /// Power off the GPIO controller
    async fn power_off(&self) -> Result<(), GpioError>;
    
    /// Check if GPIO controller is powered
    async fn is_powered(&self) -> bool;
    
    /// Configure a pin with the specified mode
    async fn configure_pin(&self, pin: u8, mode: GpioPinMode, name: Option<String>) -> Result<(), GpioError>;
    
    /// Set output pin state
    async fn set_pin(&self, pin: u8, state: GpioPinState) -> Result<(), GpioError>;
    
    /// Read pin state
    async fn read_pin(&self, pin: u8) -> Result<GpioPinState, GpioError>;
    
    /// Get all configured pins
    async fn get_pins(&self) -> Vec<GpioPin>;
    
    /// Get GPIO health information
    async fn get_health(&self) -> GpioHealth;
    
    /// Get interlock status
    async fn get_interlocks(&self) -> InterlockStatus;
    
    /// Check if system is safe to operate
    async fn is_safe_to_operate(&self) -> bool;
    
    /// Reset all interlocks (after resolving issues)
    async fn reset_interlocks(&self) -> Result<(), GpioError>;
    
    /// Set LED color for status indication
    async fn set_led(&self, led: Led, color: LedColor) -> Result<(), GpioError>;
    
    /// Get current key switch state (true = ON/operate position)
    async fn get_key_switch_state(&self) -> Result<bool, GpioError>;
    
    /// Get activation button (enable button) state
    async fn get_activation_button_active(&self) -> Result<bool, GpioError>;
    
    /// Get remaining time on activation button (in seconds)
    async fn get_activation_button_remaining_time(&self) -> Option<u64>;
    
    /// Play startup sound using system beep
    async fn play_startup_sound(&self) -> Result<(), GpioError>;
    
    /// Play radiation warning sound using system beep
    async fn play_radiation_warning(&self) -> Result<(), GpioError>;

    /// DEMO: If enable button is active, emulate opening the shutter
    /// by flipping radiation_safe to false. Returns true if changed.
    async fn emulate_open_shutter_if_enable_button_active(&self) -> Result<bool, GpioError> { Ok(false) }
    
    /// Open beam-stop (shutter) - allows X-rays through
    /// Sets radiation_safe to FALSE (beam is now open/unsafe)
    async fn open_beam_stop(&self) -> Result<(), GpioError>;
    
    /// Close beam-stop (shutter) - blocks X-rays
    /// Sets radiation_safe to TRUE (beam is now closed/safe)
    async fn close_beam_stop(&self) -> Result<(), GpioError>;
}
