use async_trait::async_trait;
use std::sync::Arc;
use std::time::Instant;
use tokio::sync::RwLock;
use tracing::info;

use super::{GpioDevice, GpioError, GpioPinMode, GpioPinState, GpioPin, GpioHealth, InterlockStatus, Led, LedColor};

/// Test implementation of GpioDevice for unit testing
pub struct TestGpio {
    state: Arc<RwLock<TestGpioState>>,
}

struct TestGpioState {
    powered: bool,
    start_time: Instant,
    pins: std::collections::HashMap<u8, GpioPin>,
    interlocks_safe: bool,
    key_switch_on: bool,
    main_status_led: LedColor,
    radiation_warning_led: LedColor,
}

impl TestGpio {
    pub fn new() -> Self {
        Self {
            state: Arc::new(RwLock::new(TestGpioState {
                powered: false,
                start_time: Instant::now(),
                pins: std::collections::HashMap::new(),
                interlocks_safe: true,
                key_switch_on: false,
                main_status_led: LedColor::Red,
                radiation_warning_led: LedColor::Green,
            })),
        }
    }

    /// Set the interlock status for testing
    pub async fn set_interlocks_safe(&self, safe: bool) {
        let mut state = self.state.write().await;
        state.interlocks_safe = safe;
    }
}

impl Default for TestGpio {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl GpioDevice for TestGpio {
    async fn power_on(&self) -> Result<(), GpioError> {
        let mut state = self.state.write().await;
        state.powered = true;
        info!("Test GPIO: Powered on");
        Ok(())
    }

    async fn power_off(&self) -> Result<(), GpioError> {
        let mut state = self.state.write().await;
        state.powered = false;
        info!("Test GPIO: Powered off");
        Ok(())
    }

    async fn is_powered(&self) -> bool {
        self.state.read().await.powered
    }

    async fn configure_pin(&self, pin: u8, mode: GpioPinMode, name: Option<String>) -> Result<(), GpioError> {
        let mut state = self.state.write().await;
        if !state.powered {
            return Err(GpioError::NotPowered);
        }

        state.pins.insert(pin, GpioPin {
            pin,
            mode,
            state: GpioPinState::Low,
            name,
        });
        info!("Test GPIO: Pin {} configured", pin);
        Ok(())
    }

    async fn set_pin(&self, pin: u8, pin_state: GpioPinState) -> Result<(), GpioError> {
        let mut state = self.state.write().await;
        if !state.powered {
            return Err(GpioError::NotPowered);
        }

        if let Some(gpio_pin) = state.pins.get_mut(&pin) {
            info!("Test GPIO: Pin {} set to {:?}", pin, pin_state);
            gpio_pin.state = pin_state;
            Ok(())
        } else {
            Err(GpioError::InvalidPin(pin))
        }
    }

    async fn read_pin(&self, pin: u8) -> Result<GpioPinState, GpioError> {
        let state = self.state.read().await;
        if !state.powered {
            return Err(GpioError::NotPowered);
        }

        state.pins.get(&pin)
            .map(|p| p.state.clone())
            .ok_or(GpioError::InvalidPin(pin))
    }

    async fn get_pins(&self) -> Vec<GpioPin> {
        let state = self.state.read().await;
        state.pins.values().cloned().collect()
    }

    async fn get_health(&self) -> GpioHealth {
        let state = self.state.read().await;
        GpioHealth {
            powered: state.powered,
            total_pins: 32,
            configured_pins: state.pins.len(),
            interlocks_status: InterlockStatus {
                emergency_stop: state.interlocks_safe,
                door_closed: state.interlocks_safe,
                radiation_safe: state.interlocks_safe,
                cooling_ok: state.interlocks_safe,
                power_ok: state.interlocks_safe,
                overall_safe: state.interlocks_safe,
            },
            uptime: state.start_time.elapsed(),
        }
    }

    async fn get_interlocks(&self) -> InterlockStatus {
        let state = self.state.read().await;
        InterlockStatus {
            emergency_stop: state.interlocks_safe,
            door_closed: state.interlocks_safe,
            radiation_safe: state.interlocks_safe,
            cooling_ok: state.interlocks_safe,
            power_ok: state.interlocks_safe,
            overall_safe: state.interlocks_safe,
        }
    }

    async fn is_safe_to_operate(&self) -> bool {
        self.state.read().await.interlocks_safe
    }

    async fn reset_interlocks(&self) -> Result<(), GpioError> {
        let mut state = self.state.write().await;
        state.interlocks_safe = true;
        info!("Test GPIO: Interlocks reset");
        Ok(())
    }
    
    async fn set_led(&self, led: Led, color: LedColor) -> Result<(), GpioError> {
        let mut state = self.state.write().await;
        if !state.powered {
            return Err(GpioError::NotPowered);
        }
        
        match led {
            Led::MainStatus => {
                state.main_status_led = color;
                info!("Test GPIO: Main Status LED set to {:?}", color);
            }
            Led::RadiationWarning => {
                state.radiation_warning_led = color;
                info!("Test GPIO: Radiation Warning LED set to {:?}", color);
            }
        }
        Ok(())
    }
    
    async fn get_key_switch_state(&self) -> Result<bool, GpioError> {
        let state = self.state.read().await;
        if !state.powered {
            return Err(GpioError::NotPowered);
        }
        Ok(state.key_switch_on)
    }
    
    async fn get_activation_button_active(&self) -> Result<bool, GpioError> {
        let state = self.state.read().await;
        if !state.powered {
            return Err(GpioError::NotPowered);
        }
        Ok(false) // Test GPIO has no activation button
    }
    
    async fn get_activation_button_remaining_time(&self) -> Option<u64> {
        None // Test GPIO has no activation button timer
    }
    
    async fn play_startup_sound(&self) -> Result<(), GpioError> {
        info!("Test GPIO: Playing startup sound (stub)");
        Ok(())
    }
    
    async fn play_radiation_warning(&self) -> Result<(), GpioError> {
        info!("Test GPIO: Playing radiation warning (stub)");
        Ok(())
    }
    
    async fn open_beam_stop(&self) -> Result<(), GpioError> {
        info!("Test GPIO: Beam-stop OPENED");
        Ok(())
    }
    
    async fn close_beam_stop(&self) -> Result<(), GpioError> {
        info!("Test GPIO: Beam-stop CLOSED");
        Ok(())
    }

    async fn emulate_open_shutter_if_enable_button_active(&self) -> Result<bool, GpioError> {
        // No-op in tests
        Ok(false)
    }
}

impl TestGpio {
    /// Set key switch state for testing
    pub async fn set_key_switch(&self, on: bool) {
        let mut state = self.state.write().await;
        state.key_switch_on = on;
    }
    
    /// Get current LED states for testing
    pub async fn get_led_state(&self, led: Led) -> LedColor {
        let state = self.state.read().await;
        match led {
            Led::MainStatus => state.main_status_led,
            Led::RadiationWarning => state.radiation_warning_led,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[tokio::test]
    async fn test_key_switch_initial_state() {
        let gpio = TestGpio::new();
        gpio.power_on().await.unwrap();
        
        // Key switch should be OFF initially
        let state = gpio.get_key_switch_state().await.unwrap();
        assert!(!state, "Key switch should be OFF initially");
    }
    
    #[tokio::test]
    async fn test_key_switch_toggle() {
        let gpio = TestGpio::new();
        gpio.power_on().await.unwrap();
        
        // Turn key switch ON
        gpio.set_key_switch(true).await;
        let state = gpio.get_key_switch_state().await.unwrap();
        assert!(state, "Key switch should be ON");
        
        // Turn key switch OFF
        gpio.set_key_switch(false).await;
        let state = gpio.get_key_switch_state().await.unwrap();
        assert!(!state, "Key switch should be OFF");
    }
    
    #[tokio::test]
    async fn test_key_switch_requires_power() {
        let gpio = TestGpio::new();
        
        // Should fail when not powered
        let result = gpio.get_key_switch_state().await;
        assert!(result.is_err(), "Should fail when GPIO not powered");
    }
    
    #[tokio::test]
    async fn test_led_control() {
        let gpio = TestGpio::new();
        gpio.power_on().await.unwrap();
        
        // Test main status LED
        gpio.set_led(Led::MainStatus, LedColor::Red).await.unwrap();
        let state = gpio.get_led_state(Led::MainStatus).await;
        assert_eq!(state, LedColor::Red);
        
        gpio.set_led(Led::MainStatus, LedColor::Orange).await.unwrap();
        let state = gpio.get_led_state(Led::MainStatus).await;
        assert_eq!(state, LedColor::Orange);
        
        gpio.set_led(Led::MainStatus, LedColor::Green).await.unwrap();
        let state = gpio.get_led_state(Led::MainStatus).await;
        assert_eq!(state, LedColor::Green);
    }
    
    #[tokio::test]
    async fn test_radiation_warning_led() {
        let gpio = TestGpio::new();
        gpio.power_on().await.unwrap();
        
        // Test radiation warning LED
        gpio.set_led(Led::RadiationWarning, LedColor::Green).await.unwrap();
        let state = gpio.get_led_state(Led::RadiationWarning).await;
        assert_eq!(state, LedColor::Green, "Should be safe (green)");
        
        gpio.set_led(Led::RadiationWarning, LedColor::Orange).await.unwrap();
        let state = gpio.get_led_state(Led::RadiationWarning).await;
        assert_eq!(state, LedColor::Orange, "Should be caution (orange)");
        
        gpio.set_led(Led::RadiationWarning, LedColor::Red).await.unwrap();
        let state = gpio.get_led_state(Led::RadiationWarning).await;
        assert_eq!(state, LedColor::Red, "Should be active radiation (red)");
    }
    
    #[tokio::test]
    async fn test_led_requires_power() {
        let gpio = TestGpio::new();
        
        // Should fail when not powered
        let result = gpio.set_led(Led::MainStatus, LedColor::Green).await;
        assert!(result.is_err(), "Should fail when GPIO not powered");
    }
    
    #[tokio::test]
    async fn test_sound_generation() {
        let gpio = TestGpio::new();
        gpio.power_on().await.unwrap();
        
        // Test startup sound (should not error)
        gpio.play_startup_sound().await.unwrap();
        
        // Test radiation warning sound (should not error)
        gpio.play_radiation_warning().await.unwrap();
    }
    
    #[tokio::test]
    async fn test_initial_led_states() {
        let gpio = TestGpio::new();
        
        // Check initial LED states
        let main_status = gpio.get_led_state(Led::MainStatus).await;
        assert_eq!(main_status, LedColor::Red, "Main status should be RED (locked) initially");
        
        let radiation = gpio.get_led_state(Led::RadiationWarning).await;
        assert_eq!(radiation, LedColor::Green, "Radiation warning should be GREEN (safe) initially");
    }
}
