use async_trait::async_trait;
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Instant;
use tokio::sync::RwLock;
use tracing::info;

type StateNotificationSender = tokio::sync::broadcast::Sender<crate::grpc::hub::v1::StateChangeNotification>;

use super::{GpioDevice, GpioError, GpioPinMode, GpioPinState, GpioPin, GpioHealth, InterlockStatus, Led, LedColor};

/// DEMO implementation of GpioDevice with interlock emulation
pub struct DemoGpio {
    pub state: Arc<RwLock<DemoGpioState>>,
    pub enable_button_timeout_secs: u32,
    notifier: Arc<RwLock<Option<StateNotificationSender>>>,
}

pub struct DemoGpioState {
    pub powered: bool,
    #[allow(dead_code)]
    pub start_time: Instant,
    pub pins: HashMap<u8, GpioPin>,
    
    // Simulated pin states (DEMO: virtual hardware, Real: read from GPIO card)
    pub simulated_pin_states: HashMap<u8, GpioPinState>,
    
    // Interlock states (derived from pins 1-5 by input watchdog)
    pub emergency_stop_ok: bool,     // Pin 1: true = released/safe, false = pressed/fault
    pub door_closed_ok: bool,        // Pin 2: true = closed/safe, false = open/fault
    pub radiation_safe_input: bool,  // Pin 3: INPUT sensor feedback from beam-stop
    pub cooling_ok: bool,            // Pin 4: true = OK/safe, false = fault
    pub power_ok: bool,              // Pin 5: true = OK/safe, false = fault
    
    // Control outputs
    pub beam_stop_output: bool,      // Pin 6 OUTPUT: Command to beam-stop actuator (true=close, false=open)
    
    // LED states (output pins 10-14)
    pub radiation_led: LedColor,     // Pin 10: Red/Orange/Green/Off
    pub door_led: bool,              // Pin 11: true = Green (closed), false = Red (open)
    pub key_led: bool,               // Pin 12: true = Green (ON), false = Red (OFF)
    pub power_led: bool,             // Pin 13: true = Green (OK), false = Red (fault)
    pub cooling_led: bool,           // Pin 14: true = Green (OK), false = Red (fault)
    
    // Key switch state (input pin 7, derived by input watchdog)
    pub key_switch_on: bool,         // true = ON/operate, false = OFF/locked
    
    // Activation button state (software timer, triggered by Pin 8 edge)
    pub activation_button_active: bool,
    pub activation_button_expires_at: Option<Instant>,
    pub last_activation_button_state: GpioPinState,  // For edge detection
}

impl DemoGpio {
    pub fn new() -> Self {
        Self::with_config(20) // Default 20 seconds
    }
    
    pub fn with_config(enable_button_timeout_secs: u32) -> Self {
        let mut pins = HashMap::new();
        
        // Pre-configure some interlock pins
        pins.insert(1, GpioPin {
            pin: 1,
            mode: GpioPinMode::Input,
            state: GpioPinState::High, // Emergency stop not pressed
            name: Some("Emergency Stop".to_string()),
        });
        
        pins.insert(2, GpioPin {
            pin: 2,
            mode: GpioPinMode::Input,
            state: GpioPinState::High, // Door closed
            name: Some("Door Closed".to_string()),
        });
        
        pins.insert(3, GpioPin {
            pin: 3,
            mode: GpioPinMode::Input,
            state: GpioPinState::High, // Radiation safe sensor input
            name: Some("Radiation Safe Input".to_string()),
        });
        
        pins.insert(4, GpioPin {
            pin: 4,
            mode: GpioPinMode::Input,
            state: GpioPinState::High, // Cooling OK
            name: Some("Cooling OK".to_string()),
        });
        
        pins.insert(5, GpioPin {
            pin: 5,
            mode: GpioPinMode::Input,
            state: GpioPinState::High, // Power OK
            name: Some("Power OK".to_string()),
        });
        
        // Beam-stop control output
        pins.insert(6, GpioPin {
            pin: 6,
            mode: GpioPinMode::Output,
            state: GpioPinState::High, // Beam-stop closed (safe) initially
            name: Some("Beam-Stop Control".to_string()),
        });
        
        // Key switch and activation button inputs
        pins.insert(7, GpioPin {
            pin: 7,
            mode: GpioPinMode::Input,
            state: GpioPinState::Low, // Key switch OFF initially
            name: Some("Key Switch".to_string()),
        });
        
        pins.insert(8, GpioPin {
            pin: 8,
            mode: GpioPinMode::Input,
            state: GpioPinState::Low, // Activation button not pressed
            name: Some("Activation Button".to_string()),
        });
        
        // LED output pins
        pins.insert(10, GpioPin {
            pin: 10,
            mode: GpioPinMode::Output,
            state: GpioPinState::Low, // Radiation warning LED
            name: Some("Radiation Warning LED".to_string()),
        });
        
        pins.insert(11, GpioPin {
            pin: 11,
            mode: GpioPinMode::Output,
            state: GpioPinState::Low, // Door status LED
            name: Some("Door Status LED".to_string()),
        });
        
        pins.insert(12, GpioPin {
            pin: 12,
            mode: GpioPinMode::Output,
            state: GpioPinState::Low, // Key switch LED
            name: Some("Key Switch LED".to_string()),
        });
        
        pins.insert(13, GpioPin {
            pin: 13,
            mode: GpioPinMode::Output,
            state: GpioPinState::Low, // Power status LED
            name: Some("Power Status LED".to_string()),
        });
        
        pins.insert(14, GpioPin {
            pin: 14,
            mode: GpioPinMode::Output,
            state: GpioPinState::Low, // Cooling status LED
            name: Some("Cooling Status LED".to_string()),
        });
        
        // Initialize simulated pin states (virtual hardware layer)
        let mut simulated_pins = HashMap::new();
        simulated_pins.insert(1, GpioPinState::High);  // E-stop released
        simulated_pins.insert(2, GpioPinState::High);  // Door closed
        simulated_pins.insert(3, GpioPinState::High);  // Radiation safe
        simulated_pins.insert(4, GpioPinState::High);  // Cooling OK
        simulated_pins.insert(5, GpioPinState::High);  // Power OK
        simulated_pins.insert(7, GpioPinState::Low);   // Key switch OFF
        simulated_pins.insert(8, GpioPinState::Low);   // Activation button not pressed

        let state = Arc::new(RwLock::new(DemoGpioState {
            powered: true,   // GPIO hardware auto-starts with computer
            start_time: Instant::now(),
            pins,
            simulated_pin_states: simulated_pins,
            // Interlock states (will be updated by watchdog)
            emergency_stop_ok: true,
            door_closed_ok: true,
            radiation_safe_input: true,
            cooling_ok: true,
            power_ok: true,
            beam_stop_output: true,      // Command beam-stop CLOSED (safe)
            // LED states
            radiation_led: LedColor::Green,  // Safe initially
            door_led: true,                  // Green (closed)
            key_led: false,                  // Red (OFF)
            power_led: true,                 // Green (OK)
            cooling_led: true,               // Green (OK)
            // Key switch
            key_switch_on: false,            // OFF initially
            // Activation button
            activation_button_active: false,
            activation_button_expires_at: None,
            last_activation_button_state: GpioPinState::Low,
        }));
        
        // Spawn input watchdog task to poll simulated pins
        let state_clone = state.clone();
        tokio::spawn(async move {
            DemoGpio::input_watchdog_task(state_clone).await;
        });
        
        // Spawn beam-stop watchdog (monitors output->input feedback)
        let state_clone = state.clone();
        tokio::spawn(async move {
            DemoGpio::beam_stop_watchdog(state_clone).await;
        });
        
        Self {
            state,
            enable_button_timeout_secs,
            notifier: Arc::new(RwLock::new(None)),
        }
    }
}

impl Default for DemoGpio {
    fn default() -> Self {
        Self::new()
    }
}

impl DemoGpio {
    /// Input watchdog task that monitors simulated pin states and updates interlock fields
    /// Polls every 100ms (demo) to emulate real GPIO hardware input monitoring
    async fn input_watchdog_task(state: Arc<RwLock<DemoGpioState>>) {
        loop {
            tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;
            
            let mut s = state.write().await;
            if !s.powered {
                tokio::time::sleep(tokio::time::Duration::from_secs(1)).await;
                continue;
            }
            
            // Read simulated pin states and update interlock fields
            if let Some(&pin_state) = s.simulated_pin_states.get(&1) {
                s.emergency_stop_ok = pin_state == GpioPinState::High;
            }
            if let Some(&pin_state) = s.simulated_pin_states.get(&2) {
                s.door_closed_ok = pin_state == GpioPinState::High;
            }
            if let Some(&pin_state) = s.simulated_pin_states.get(&3) {
                s.radiation_safe_input = pin_state == GpioPinState::High;
            }
            if let Some(&pin_state) = s.simulated_pin_states.get(&4) {
                s.cooling_ok = pin_state == GpioPinState::High;
            }
            if let Some(&pin_state) = s.simulated_pin_states.get(&5) {
                s.power_ok = pin_state == GpioPinState::High;
            }
            
            // Update LED states based on interlock conditions
            s.radiation_led = if s.radiation_safe_input { LedColor::Green } else { LedColor::Red };
            s.door_led = s.door_closed_ok;
            s.power_led = s.power_ok;
            s.cooling_led = s.cooling_ok;
            
            // Handle key switch (pin 7)
            if let Some(&pin_state) = s.simulated_pin_states.get(&7) {
                s.key_switch_on = pin_state == GpioPinState::High;
                s.key_led = s.key_switch_on;
            }
            
            // Handle activation button (pin 8) with edge detection
            if let Some(&pin_state) = s.simulated_pin_states.get(&8) {
                // Detect rising edge (button press)
                if pin_state == GpioPinState::High && s.last_activation_button_state == GpioPinState::Low {
                    // Button just pressed - activate for configured duration
                    s.activation_button_active = true;
                    // Note: timeout duration is managed by DemoGpio::enable_button_timeout_secs
                    info!("DEMO GPIO: 🔘 Activation button pressed (edge detected)");
                }
                s.last_activation_button_state = pin_state;
            }
        }
    }
    
    /// Watchdog task that monitors beam_stop_output and updates radiation_safe_input
    /// Simulates mechanical delay of physical beam-stop actuator
    async fn beam_stop_watchdog(state: Arc<RwLock<DemoGpioState>>) {
        let mut last_output_state = {
            let s = state.read().await;
            s.beam_stop_output
        };
        
        loop {
            tokio::time::sleep(tokio::time::Duration::from_millis(50)).await;
            
            let current_output_state = {
                let s = state.read().await;
                if !s.powered {
                    tokio::time::sleep(tokio::time::Duration::from_secs(1)).await;
                    continue;
                }
                s.beam_stop_output
            };
            
            // Detect state change
            if current_output_state != last_output_state {
                last_output_state = current_output_state;
                
                // Simulate mechanical delay (100-300ms for beam-stop movement)
                let delay_ms = if current_output_state { 150 } else { 200 }; // Close faster than open
                info!("DEMO GPIO: 🔄 Beam-stop output changed to {} - simulating {}ms mechanical delay",
                      if current_output_state { "CLOSE" } else { "OPEN" }, delay_ms);
                tokio::time::sleep(tokio::time::Duration::from_millis(delay_ms)).await;
                
                // Update input sensor to reflect actual position
                {
                    let mut s = state.write().await;
                    s.radiation_safe_input = current_output_state;
                }
                
                info!("DEMO GPIO: ✅ Radiation safe input updated to {} (beam-stop movement complete)",
                      if current_output_state { "SAFE" } else { "UNSAFE" });
            }
        }
    }
    
    /// Set the notification sender for state change broadcasts (async)
    pub async fn set_notifier(&self, notifier: StateNotificationSender) {
        let mut n = self.notifier.write().await;
        *n = Some(notifier);
    }
    
    /// Set the notification sender for state change broadcasts (synchronous)
    pub fn set_notifier_sync(&self, notifier: StateNotificationSender) {
        let mut n = self.notifier.blocking_write();
        *n = Some(notifier);
    }
    
    /// Emit a state change notification (best-effort)
    fn emit_notification(&self, component: &str, change_type: &str) {
        // Try to get notifier without blocking
        match self.notifier.try_read() {
            Ok(notifier_lock) => {
                match notifier_lock.as_ref() {
                    Some(tx) => {
                        let notification = crate::grpc::hub::v1::StateChangeNotification {
                            component: component.to_string(),
                            change_type: change_type.to_string(),
                            timestamp: Some(prost_types::Timestamp {
                                seconds: chrono::Utc::now().timestamp(),
                                nanos: 0,
                            }),
                        };
                        match tx.send(notification) {
                            Ok(count) => {
                                tracing::debug!("📢 Emitted notification: {} - {} (subscribers: {})", component, change_type, count);
                            }
                            Err(_) => {
                                tracing::warn!("Failed to send notification: no subscribers");
                            }
                        }
                    }
                    None => {
                        tracing::warn!("⚠️  Notifier not set - cannot emit notification");
                    }
                }
            }
            Err(_) => {
                tracing::warn!("⚠️  Could not acquire notifier lock");
            }
        }
    }
}

#[async_trait]
impl GpioDevice for DemoGpio {
    async fn power_on(&self) -> Result<(), GpioError> {
        let mut state = self.state.write().await;
        if state.powered {
            return Ok(());
        }
        
        state.powered = true;
        info!("DEMO GPIO: Powered on with {} configured pins", state.pins.len());
        Ok(())
    }

    async fn power_off(&self) -> Result<(), GpioError> {
        let mut state = self.state.write().await;
        state.powered = false;
        
        // Set all output pins to low when powering off
        for pin in state.pins.values_mut() {
            if pin.mode == GpioPinMode::Output {
                pin.state = GpioPinState::Low;
            }
        }
        
        info!("DEMO GPIO: Powered off");
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
        
        if pin > 32 {
            return Err(GpioError::InvalidPin(pin));
        }

        let initial_state = match mode {
            GpioPinMode::Output => GpioPinState::Low,
            GpioPinMode::Input | GpioPinMode::InputPullUp => GpioPinState::High,
            GpioPinMode::InputPullDown => GpioPinState::Low,
        };

        info!("DEMO GPIO: Pin {} configured as {:?}", pin, mode);
        
        state.pins.insert(pin, GpioPin {
            pin,
            mode,
            state: initial_state,
            name,
        });
        Ok(())
    }

    async fn set_pin(&self, pin: u8, state_val: GpioPinState) -> Result<(), GpioError> {
        let mut state = self.state.write().await;
        
        if !state.powered {
            return Err(GpioError::NotPowered);
        }

        match state.pins.get_mut(&pin) {
            Some(gpio_pin) => {
                if gpio_pin.mode != GpioPinMode::Output {
                    return Err(GpioError::PinNotConfigured(pin));
                }
                info!("DEMO GPIO: Pin {} set to {:?}", pin, state_val);
                gpio_pin.state = state_val;
                Ok(())
            }
            None => Err(GpioError::InvalidPin(pin)),
        }
    }

    async fn read_pin(&self, pin: u8) -> Result<GpioPinState, GpioError> {
        let state = self.state.read().await;
        
        if !state.powered {
            return Err(GpioError::NotPowered);
        }

        match state.pins.get(&pin) {
            Some(gpio_pin) => Ok(gpio_pin.state.clone()),
            None => Err(GpioError::InvalidPin(pin)),
        }
    }

    async fn get_pins(&self) -> Vec<GpioPin> {
        let state = self.state.read().await;
        state.pins.values().cloned().collect()
    }

    async fn get_health(&self) -> GpioHealth {
        let state = self.state.read().await;
        GpioHealth {
            powered: state.powered,
            total_pins: 32, // Simulate 32-pin controller
            configured_pins: state.pins.len(),
            interlocks_status: InterlockStatus {
                emergency_stop: state.emergency_stop_ok,
                door_closed: state.door_closed_ok,
                radiation_safe: state.radiation_safe_input,  // Use INPUT sensor
                cooling_ok: state.cooling_ok,
                power_ok: state.power_ok,
                overall_safe: state.emergency_stop_ok && state.door_closed_ok &&
                             state.radiation_safe_input && state.cooling_ok && state.power_ok,
            },
            uptime: state.start_time.elapsed(),
        }
    }

    async fn get_interlocks(&self) -> InterlockStatus {
        let state = self.state.read().await;
        InterlockStatus {
            emergency_stop: state.emergency_stop_ok,
            door_closed: state.door_closed_ok,
            radiation_safe: state.radiation_safe_input,  // Use INPUT sensor
            cooling_ok: state.cooling_ok,
            power_ok: state.power_ok,
            overall_safe: state.emergency_stop_ok && state.door_closed_ok &&
                         state.radiation_safe_input && state.cooling_ok && state.power_ok,
        }
    }

    async fn is_safe_to_operate(&self) -> bool {
        let interlocks = self.get_interlocks().await;
        interlocks.overall_safe
    }

    async fn reset_interlocks(&self) -> Result<(), GpioError> {
        let mut state = self.state.write().await;
        
        if !state.powered {
            return Err(GpioError::NotPowered);
        }

        // Reset all interlock conditions to safe state
        state.emergency_stop_ok = true;     // E-stop released (safe)
        state.door_closed_ok = true;
        state.radiation_safe_input = true;  // Sensor reads safe
        state.beam_stop_output = true;      // Command beam-stop closed
        state.cooling_ok = true;
        state.power_ok = true;
        
        info!("DEMO GPIO: All interlocks reset to safe state");
        Ok(())
    }
    
    async fn set_led(&self, led: Led, color: LedColor) -> Result<(), GpioError> {
        let mut state = self.state.write().await;
        
        if !state.powered {
            return Err(GpioError::NotPowered);
        }
        
        match led {
            Led::MainStatus => {
                // Map MainStatus to Key LED (for backwards compatibility)
                state.key_led = color != LedColor::Red && color != LedColor::Off;
                info!("DEMO GPIO: Main Status LED (Key) set to {:?}", color);
            }
            Led::RadiationWarning => {
                // Map RadiationWarning to radiation LED
                state.radiation_led = color;
                info!("DEMO GPIO: Radiation Warning LED set to {:?}", color);
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
        let mut state = self.state.write().await;
        
        if !state.powered {
            return Err(GpioError::NotPowered);
        }
        
        // Expire if past deadline
        if let Some(expires_at) = state.activation_button_expires_at {
            if Instant::now() >= expires_at {
                state.activation_button_active = false;
                state.activation_button_expires_at = None;
                drop(state); // Release lock before emitting
                info!("DEMO GPIO: ⏱️  Activation button EXPIRED (queried)");
                self.emit_notification("GPIO", "ENABLE_BUTTON_DEACTIVATED");
                return Ok(false);
            }
        }
        
        Ok(state.activation_button_active)
    }
    
    async fn play_startup_sound(&self) -> Result<(), GpioError> {
        info!("DEMO GPIO: 🔊 Playing startup sound (3 ascending beeps: 500Hz, 750Hz, 1000Hz)");
        
        // TODO: Implement actual sound generation when industrial hardware is available
        // For now, just log the event. In production, this will interface with
        // industrial PC speaker or piezo buzzer via GPIO or dedicated sound hardware.
        
        Ok(())
    }
    
    async fn play_radiation_warning(&self) -> Result<(), GpioError> {
        info!("DEMO GPIO: ⚠️  Playing radiation warning sound (pulsed 1500Hz warning tone)");
        
        // TODO: Implement actual sound generation when industrial hardware is available
        // For now, just log the event. In production, this will interface with
        // industrial PC speaker or piezo buzzer via GPIO or dedicated sound hardware.
        
        Ok(())
    }

    async fn emulate_open_shutter_if_enable_button_active(&self) -> Result<bool, GpioError> {
        self.emulate_open_shutter_if_enable_button_active_impl().await
    }

    async fn get_activation_button_remaining_time(&self) -> Option<u64> {
        let state = self.state.read().await;

        if let Some(expires_at) = state.activation_button_expires_at {
            let now = Instant::now();
            if now < expires_at {
                let remaining = expires_at.duration_since(now);
                return Some(remaining.as_secs());
            }
        }

        None
    }
    
    /// Open beam-stop (shutter) - allows X-rays through
    /// Sets beam_stop_output to FALSE - watchdog will update radiation_safe_input after mechanical delay
    async fn open_beam_stop(&self) -> Result<(), GpioError> {
        let mut state = self.state.write().await;
        if !state.powered {
            return Err(GpioError::NotPowered);
        }
        state.beam_stop_output = false;  // Command: OPEN beam-stop
        drop(state);
        info!("DEMO GPIO: 🔓 Beam-stop OPEN command sent (watchdog will update sensor after mechanical delay)");
        self.emit_notification("GPIO", "BEAM_STOP_OPEN_COMMANDED");
        self.set_led(super::Led::RadiationWarning, super::LedColor::Orange).await.ok(); // Orange = transitioning
        Ok(())
    }
    
    /// Close beam-stop (shutter) - blocks X-rays
    /// Sets beam_stop_output to TRUE - watchdog will update radiation_safe_input after mechanical delay
    async fn close_beam_stop(&self) -> Result<(), GpioError> {
        let mut state = self.state.write().await;
        if !state.powered {
            return Err(GpioError::NotPowered);
        }
        state.beam_stop_output = true;  // Command: CLOSE beam-stop
        drop(state);
        info!("DEMO GPIO: 🔒 Beam-stop CLOSE command sent (watchdog will update sensor after mechanical delay)");
        self.emit_notification("GPIO", "BEAM_STOP_CLOSE_COMMANDED");
        self.set_led(super::Led::RadiationWarning, super::LedColor::Orange).await.ok(); // Orange = transitioning
        Ok(())
    }
}

impl DemoGpio {
    /// Emulate opening shutter if activation button is active
    pub async fn emulate_open_shutter_if_enable_button_active_impl(&self) -> Result<bool, GpioError> {
        // If activation button is active, flip radiation_safe to false
        if self.is_activation_button_active().await {
            self.set_radiation_safe(false).await?;
            self.set_led(super::Led::RadiationWarning, super::LedColor::Red).await.ok();
            return Ok(true);
        }
        Ok(false)
    }

    /// DEMO ONLY: Set emergency stop state
    /// @param pressed: true = button PRESSED (unsafe), false = button RELEASED (safe)
    pub async fn set_emergency_stop(&self, pressed: bool) -> Result<(), GpioError> {
        let mut state = self.state.write().await;
        if !state.powered {
            return Err(GpioError::NotPowered);
        }
        // Simulate hardware input: pressed=Low, released=High
        let pin_state = if pressed { GpioPinState::Low } else { GpioPinState::High };
        state.simulated_pin_states.insert(1, pin_state);
        drop(state);
        info!("DEMO GPIO: Emergency stop set to {} (simulated pin 1)", if pressed { "PRESSED" } else { "RELEASED" });
        self.emit_notification("GPIO", "INTERLOCK_CHANGED");
        Ok(())
    }
    
    /// DEMO ONLY: Set door closed state
    pub async fn set_door_closed(&self, closed: bool) -> Result<(), GpioError> {
        let mut state = self.state.write().await;
        if !state.powered {
            return Err(GpioError::NotPowered);
        }
        // Simulate hardware input: closed=High, open=Low
        let pin_state = if closed { GpioPinState::High } else { GpioPinState::Low };
        state.simulated_pin_states.insert(2, pin_state);
        drop(state);
        info!("DEMO GPIO: Door set to {} (simulated pin 2)", if closed { "CLOSED" } else { "OPEN" });
        self.emit_notification("GPIO", "INTERLOCK_CHANGED");
        Ok(())
    }
    
    /// DEMO ONLY: Set radiation safe INPUT state (sensor feedback)
    /// NOTE: In normal operation, use open_beam_stop/close_beam_stop which control the OUTPUT
    pub async fn set_radiation_safe(&self, safe: bool) -> Result<(), GpioError> {
        let mut state = self.state.write().await;
        if !state.powered {
            return Err(GpioError::NotPowered);
        }
        // Simulate hardware input: safe=High, unsafe=Low
        let pin_state = if safe { GpioPinState::High } else { GpioPinState::Low };
        state.simulated_pin_states.insert(3, pin_state);
        drop(state);
        info!("DEMO GPIO: Radiation INPUT set to {} (simulated pin 3)", if safe { "SAFE" } else { "UNSAFE" });
        self.emit_notification("GPIO", "INTERLOCK_CHANGED");
        Ok(())
    }
    
    /// DEMO ONLY: Set cooling OK state  
    pub async fn set_cooling_ok(&self, ok: bool) -> Result<(), GpioError> {
        let mut state = self.state.write().await;
        if !state.powered {
            return Err(GpioError::NotPowered);
        }
        // Simulate hardware input: OK=High, fault=Low
        let pin_state = if ok { GpioPinState::High } else { GpioPinState::Low };
        state.simulated_pin_states.insert(4, pin_state);
        drop(state);
        info!("DEMO GPIO: Cooling set to {} (simulated pin 4)", if ok { "OK" } else { "FAULT" });
        self.emit_notification("GPIO", "INTERLOCK_CHANGED");
        Ok(())
    }
    
    /// DEMO ONLY: Set power OK state
    pub async fn set_power_ok(&self, ok: bool) -> Result<(), GpioError> {
        let mut state = self.state.write().await;
        if !state.powered {
            return Err(GpioError::NotPowered);
        }
        // Simulate hardware input: OK=High, fault=Low
        let pin_state = if ok { GpioPinState::High } else { GpioPinState::Low };
        state.simulated_pin_states.insert(5, pin_state);
        drop(state);
        info!("DEMO GPIO: Power set to {} (simulated pin 5)", if ok { "OK" } else { "FAULT" });
        self.emit_notification("GPIO", "INTERLOCK_CHANGED");
        Ok(())
    }
    
    /// DEMO ONLY: Set all interlocks to safe (armed) state
    pub async fn arm_all_interlocks(&self) -> Result<(), GpioError> {
        let mut state = self.state.write().await;
        if !state.powered {
            return Err(GpioError::NotPowered);
        }
        // Simulate all inputs to safe states (High = safe)
        state.simulated_pin_states.insert(1, GpioPinState::High);  // E-stop released
        state.simulated_pin_states.insert(2, GpioPinState::High);  // Door closed
        state.simulated_pin_states.insert(3, GpioPinState::High);  // Radiation safe
        state.simulated_pin_states.insert(4, GpioPinState::High);  // Cooling OK
        state.simulated_pin_states.insert(5, GpioPinState::High);  // Power OK
        state.beam_stop_output = true;  // Command beam-stop closed
        info!("DEMO GPIO: ✅ All interlocks ARMED (simulated pins -> High)");
        Ok(())
    }
    
    /// DEMO ONLY: Set key switch state (async)
    pub async fn set_key_switch(&self, on: bool) -> Result<(), GpioError> {
        let mut state = self.state.write().await;
        if !state.powered {
            return Err(GpioError::NotPowered);
        }
        // Simulate hardware input: ON=High, OFF=Low
        let pin_state = if on { GpioPinState::High } else { GpioPinState::Low };
        state.simulated_pin_states.insert(7, pin_state);
        drop(state); // Release lock before emitting
        info!("DEMO GPIO: Key switch set to {} (simulated pin 7)", if on { "ON" } else { "OFF" });
        self.emit_notification("GPIO", "KEY_SWITCH_CHANGED");
        Ok(())
    }
    
    /// DEMO ONLY: Set key switch state (sync, for GUI embedding)
    pub fn set_key_switch_sync(&self, on: bool) -> Result<(), GpioError> {
        let mut state = self.state.blocking_write();
        if !state.powered {
            return Err(GpioError::NotPowered);
        }
        // Simulate hardware input: ON=High, OFF=Low
        let pin_state = if on { GpioPinState::High } else { GpioPinState::Low };
        state.simulated_pin_states.insert(7, pin_state);
        drop(state); // Release lock before emitting
        info!("DEMO GPIO: Key switch set to {} (simulated pin 7)", if on { "ON" } else { "OFF" });
        self.emit_notification("GPIO", "KEY_SWITCH_CHANGED");
        Ok(())
    }
    
    /// Get activation button status (enable button)
    pub async fn get_activation_button_status(&self) -> bool {
        let state = self.state.read().await;
        state.activation_button_active
    }
    
    /// DEMO ONLY: Set activation button state  
    pub async fn set_activation_button(&self, active: bool) -> Result<(), GpioError> {
        let mut state = self.state.write().await;
        if !state.powered {
            return Err(GpioError::NotPowered);
        }
        state.activation_button_active = active;
        info!("DEMO GPIO: Activation button set to {}", if active { "ACTIVE" } else { "INACTIVE" });
        Ok(())
    }
    
    /// Activate the enable button for configured timeout duration
    /// This allows non-safe operations (detector/motion initialization) to proceed
    pub async fn activate_enable_button(&self) -> Result<(), GpioError> {
        let mut state = self.state.write().await;
        if !state.powered {
            return Err(GpioError::NotPowered);
        }
        
        let timeout_secs = self.enable_button_timeout_secs as u64;
        state.activation_button_active = true;
        state.activation_button_expires_at = Some(Instant::now() + std::time::Duration::from_secs(timeout_secs));
        drop(state); // Release lock before emitting
        info!("DEMO GPIO: ⚡ Activation button ACTIVATED for {} seconds", timeout_secs);
        self.emit_notification("GPIO", "ENABLE_BUTTON_ACTIVATED");
        
        // Spawn background task to automatically deactivate after timeout
        let state_clone = self.state.clone();
        let notifier_clone = self.notifier.clone();
        tokio::spawn(async move {
            // Countdown every second
            for remaining in (1..=timeout_secs).rev() {
                tokio::time::sleep(tokio::time::Duration::from_secs(1)).await;
                let seconds_left = remaining - 1;
                info!("DEMO GPIO: ⏱️  Activation button countdown: {} seconds remaining", seconds_left);
                
                // Emit state notification for countdown
                if let Ok(notifier_lock) = notifier_clone.try_read() {
                    if let Some(tx) = notifier_lock.as_ref() {
                        let notification = crate::grpc::hub::v1::StateChangeNotification {
                            component: "GPIO".to_string(),
                            change_type: format!("ENABLE_BUTTON_COUNTDOWN_{}", seconds_left),
                            timestamp: Some(prost_types::Timestamp {
                                seconds: chrono::Utc::now().timestamp(),
                                nanos: 0,
                            }),
                        };
                        let _ = tx.send(notification);
                    }
                }
            }
            
            // Deactivate button after timeout
            if let Ok(mut state) = state_clone.try_write() {
                if state.activation_button_active {
                    state.activation_button_active = false;
                    state.activation_button_expires_at = None;
                    drop(state);
                    info!("DEMO GPIO: ⏱️  Activation button AUTO-EXPIRED after timeout");
                    
                    // Emit notification
                    if let Ok(notifier_lock) = notifier_clone.try_read() {
                        if let Some(tx) = notifier_lock.as_ref() {
                            let notification = crate::grpc::hub::v1::StateChangeNotification {
                                component: "GPIO".to_string(),
                                change_type: "ENABLE_BUTTON_DEACTIVATED".to_string(),
                                timestamp: Some(prost_types::Timestamp {
                                    seconds: chrono::Utc::now().timestamp(),
                                    nanos: 0,
                                }),
                            };
                            let _ = tx.send(notification);
                        }
                    }
                }
            }
        });
        
        Ok(())
    }
    
    /// Check if activation button is currently active (updates state if expired)
    pub async fn is_activation_button_active(&self) -> bool {
        let mut state = self.state.write().await;
        
        // Check if expired
        if let Some(expires_at) = state.activation_button_expires_at {
            if Instant::now() >= expires_at {
                state.activation_button_active = false;
                state.activation_button_expires_at = None;
                drop(state); // Release lock before emitting
                info!("DEMO GPIO: ⏱️  Activation button EXPIRED");
                self.emit_notification("GPIO", "ENABLE_BUTTON_DEACTIVATED");
                return false;
            }
        }
        
        state.activation_button_active
    }
}

impl DemoGpio {
    // ===== Synchronous helpers for GUI embedding =====
    pub fn set_emergency_stop_sync(&self, pressed: bool) -> Result<(), GpioError> {
        let mut state = self.state.blocking_write();
        if !state.powered { return Err(GpioError::NotPowered); }
        state.emergency_stop_ok = !pressed;  // Invert: pressed=false means ok=true
        drop(state);
        info!("DEMO GPIO: Emergency stop set to {}", if pressed { "PRESSED" } else { "RELEASED" });
        self.emit_notification("GPIO", "INTERLOCK_CHANGED");
        Ok(())
    }
    pub fn set_door_closed_sync(&self, closed: bool) -> Result<(), GpioError> {
        let mut state = self.state.blocking_write();
        if !state.powered { return Err(GpioError::NotPowered); }
        state.door_closed_ok = closed;
        drop(state);
        info!("DEMO GPIO: Door set to {}", if closed { "CLOSED" } else { "OPEN" });
        self.emit_notification("GPIO", "INTERLOCK_CHANGED");
        Ok(())
    }
    pub fn set_radiation_safe_sync(&self, safe: bool) -> Result<(), GpioError> {
        let mut state = self.state.blocking_write();
        if !state.powered { return Err(GpioError::NotPowered); }
        state.radiation_safe_input = safe;  // Use input field
        drop(state);
        info!("DEMO GPIO: Radiation INPUT set to {}", if safe { "SAFE" } else { "UNSAFE" });
        self.emit_notification("GPIO", "INTERLOCK_CHANGED");
        Ok(())
    }
    pub fn set_cooling_ok_sync(&self, ok: bool) -> Result<(), GpioError> {
        let mut state = self.state.blocking_write();
        if !state.powered { return Err(GpioError::NotPowered); }
        state.cooling_ok = ok;
        drop(state);
        info!("DEMO GPIO: Cooling set to {}", if ok { "OK" } else { "FAULT" });
        // Emit interlock change notification
        self.emit_notification("GPIO", "INTERLOCK_CHANGED");
        Ok(())
    }
    pub fn set_power_ok_sync(&self, ok: bool) -> Result<(), GpioError> {
        let mut state = self.state.blocking_write();
        if !state.powered { return Err(GpioError::NotPowered); }
        state.power_ok = ok;
        drop(state);
        info!("DEMO GPIO: Power set to {}", if ok { "OK" } else { "FAULT" });
        // Emit interlock change notification
        self.emit_notification("GPIO", "INTERLOCK_CHANGED");
        Ok(())
    }
    pub fn arm_all_interlocks_sync(&self) -> Result<(), GpioError> {
        let mut state = self.state.blocking_write();
        if !state.powered { return Err(GpioError::NotPowered); }
        state.emergency_stop_ok = true;     // E-stop released (safe)
        state.door_closed_ok = true;
        state.radiation_safe_input = true;  // Use input field
        state.beam_stop_output = true;      // Set output to closed
        state.cooling_ok = true;
        state.power_ok = true;
        info!("DEMO GPIO: ✅ All interlocks ARMED - system ready for operation");
        Ok(())
    }
    pub fn activate_enable_button_sync(&self) -> Result<(), GpioError> {
        let mut state = self.state.blocking_write();
        if !state.powered { return Err(GpioError::NotPowered); }
        let timeout_secs = self.enable_button_timeout_secs as u64;
        state.activation_button_active = true;
        state.activation_button_expires_at = Some(Instant::now() + std::time::Duration::from_secs(timeout_secs));
        drop(state);
        info!("DEMO GPIO: ⚡ Activation button ACTIVATED for {} seconds", timeout_secs);
        self.emit_notification("GPIO", "ENABLE_BUTTON_ACTIVATED");
        Ok(())
    }
    pub fn is_activation_button_active_sync(&self) -> bool {
        let mut state = self.state.blocking_write();
        if let Some(expires_at) = state.activation_button_expires_at {
            if Instant::now() >= expires_at {
                state.activation_button_active = false;
                state.activation_button_expires_at = None;
                drop(state); // Release lock before emitting
                info!("DEMO GPIO: ⏱️  Activation button EXPIRED");
                self.emit_notification("GPIO", "ENABLE_BUTTON_DEACTIVATED");
                return false;
            }
        }
        state.activation_button_active
    }
    pub fn get_activation_button_remaining_time_sync(&self) -> Option<u64> {
        let state = self.state.blocking_read();
        if let Some(expires_at) = state.activation_button_expires_at {
            let now = Instant::now();
            if now < expires_at {
                let remaining = expires_at.duration_since(now);
                return Some(remaining.as_secs());
            }
        }
        None
    }
    
    /// Synchronous: Open beam-stop (shutter)
    pub fn open_beam_stop_sync(&self) -> Result<(), GpioError> {
        let mut state = self.state.blocking_write();
        if !state.powered { return Err(GpioError::NotPowered); }
        state.beam_stop_output = false;  // Command: OPEN
        drop(state);
        info!("DEMO GPIO: 🔓 Beam-stop OPEN command sent (watchdog will update sensor)");
        self.emit_notification("GPIO", "BEAM_STOP_OPEN_COMMANDED");
        Ok(())
    }
    
    /// Synchronous: Close beam-stop (shutter)
    pub fn close_beam_stop_sync(&self) -> Result<(), GpioError> {
        let mut state = self.state.blocking_write();
        if !state.powered { return Err(GpioError::NotPowered); }
        state.beam_stop_output = true;  // Command: CLOSE
        drop(state);
        info!("DEMO GPIO: 🔒 Beam-stop CLOSE command sent (watchdog will update sensor)");
        self.emit_notification("GPIO", "BEAM_STOP_CLOSE_COMMANDED");
        Ok(())
    }
}
