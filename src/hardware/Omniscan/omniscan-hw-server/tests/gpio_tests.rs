use omniscan_hw_server::devices::gpio::{
    DemoGpio, GpioDevice, GpioError, GpioPinMode, GpioPinState, Led, LedColor, TestGpio,
};

#[tokio::test]
async fn test_gpio_power_on_off() {
    let gpio = TestGpio::new();

    // Initially powered off
    assert!(!gpio.is_powered().await);

    // Power on
    gpio.power_on().await.unwrap();
    assert!(gpio.is_powered().await);

    // Power off
    gpio.power_off().await.unwrap();
    assert!(!gpio.is_powered().await);
}

#[tokio::test]
async fn test_gpio_configure_pin_requires_power() {
    let gpio = TestGpio::new();

    // Should fail when not powered
    let result = gpio
        .configure_pin(1, GpioPinMode::Output, Some("Test Pin".to_string()))
        .await;
    assert!(matches!(result, Err(GpioError::NotPowered)));

    // Should succeed when powered
    gpio.power_on().await.unwrap();
    gpio.configure_pin(1, GpioPinMode::Output, Some("Test Pin".to_string()))
        .await
        .unwrap();

    // Verify pin is configured
    let pins = gpio.get_pins().await;
    assert_eq!(pins.len(), 1);
    assert_eq!(pins[0].pin, 1);
    assert_eq!(pins[0].mode, GpioPinMode::Output);
}

#[tokio::test]
async fn test_gpio_set_pin_requires_output_mode() {
    let gpio = TestGpio::new();
    gpio.power_on().await.unwrap();

    // Configure as input
    gpio.configure_pin(1, GpioPinMode::Input, None)
        .await
        .unwrap();

    // Should fail to set pin since it's input mode
    let result = gpio.set_pin(1, GpioPinState::High).await;
    assert!(matches!(result, Err(GpioError::PinNotConfigured(_))));

    // Configure as output
    gpio.configure_pin(2, GpioPinMode::Output, None)
        .await
        .unwrap();

    // Should succeed
    gpio.set_pin(2, GpioPinState::High).await.unwrap();
    let state = gpio.read_pin(2).await.unwrap();
    assert_eq!(state, GpioPinState::High);
}

#[tokio::test]
async fn test_gpio_read_pin() {
    let gpio = TestGpio::new();
    gpio.power_on().await.unwrap();

    gpio.configure_pin(1, GpioPinMode::Output, None)
        .await
        .unwrap();

    // Set high and read
    gpio.set_pin(1, GpioPinState::High).await.unwrap();
    assert_eq!(gpio.read_pin(1).await.unwrap(), GpioPinState::High);

    // Set low and read
    gpio.set_pin(1, GpioPinState::Low).await.unwrap();
    assert_eq!(gpio.read_pin(1).await.unwrap(), GpioPinState::Low);
}

#[tokio::test]
async fn test_gpio_invalid_pin() {
    let gpio = DemoGpio::new();
    gpio.power_on().await.unwrap();

    // Try to configure invalid pin (> 32)
    let result = gpio.configure_pin(33, GpioPinMode::Output, None).await;
    assert!(matches!(result, Err(GpioError::InvalidPin(33))));

    // Try to read unconfigured pin
    let result = gpio.read_pin(99).await;
    assert!(matches!(result, Err(GpioError::InvalidPin(99))));
}

#[tokio::test]
async fn test_gpio_interlocks() {
    let gpio = TestGpio::new();
    gpio.power_on().await.unwrap();

    // Initially safe
    assert!(gpio.is_safe_to_operate().await);

    // Set interlocks unsafe
    gpio.set_interlocks_safe(false).await;
    assert!(!gpio.is_safe_to_operate().await);

    // Get interlock status
    let status = gpio.get_interlocks().await;
    assert!(!status.emergency_stop);
    assert!(!status.overall_safe);

    // Reset interlocks
    gpio.reset_interlocks().await.unwrap();
    assert!(gpio.is_safe_to_operate().await);
}

#[tokio::test]
async fn test_gpio_health_info() {
    let gpio = TestGpio::new();

    let health = gpio.get_health().await;
    assert!(!health.powered);
    assert_eq!(health.total_pins, 32);

    gpio.power_on().await.unwrap();

    let health = gpio.get_health().await;
    assert!(health.powered);
    assert!(health.interlocks_status.overall_safe);
}

#[tokio::test]
async fn test_gpio_led_control() {
    let gpio = TestGpio::new();

    // Should fail when not powered
    let result = gpio.set_led(Led::MainStatus, LedColor::Green).await;
    assert!(matches!(result, Err(GpioError::NotPowered)));

    gpio.power_on().await.unwrap();

    // Set main status LED
    gpio.set_led(Led::MainStatus, LedColor::Green)
        .await
        .unwrap();
    assert_eq!(gpio.get_led_state(Led::MainStatus).await, LedColor::Green);

    gpio.set_led(Led::MainStatus, LedColor::Red).await.unwrap();
    assert_eq!(gpio.get_led_state(Led::MainStatus).await, LedColor::Red);

    // Set radiation warning LED
    gpio.set_led(Led::RadiationWarning, LedColor::Orange)
        .await
        .unwrap();
    assert_eq!(
        gpio.get_led_state(Led::RadiationWarning).await,
        LedColor::Orange
    );
}

#[tokio::test]
async fn test_gpio_key_switch() {
    let gpio = TestGpio::new();
    gpio.power_on().await.unwrap();

    // Initially OFF
    assert!(!gpio.get_key_switch_state().await.unwrap());

    // Turn ON
    gpio.set_key_switch(true).await;
    assert!(gpio.get_key_switch_state().await.unwrap());

    // Turn OFF
    gpio.set_key_switch(false).await;
    assert!(!gpio.get_key_switch_state().await.unwrap());
}

#[tokio::test]
async fn test_gpio_activation_button() {
    let gpio = TestGpio::new();
    gpio.power_on().await.unwrap();

    // Test GPIO doesn't have activation button functionality
    assert!(!gpio.get_activation_button_active().await.unwrap());
    assert!(gpio.get_activation_button_remaining_time().await.is_none());
}

#[tokio::test]
async fn test_gpio_sound_generation() {
    let gpio = TestGpio::new();
    gpio.power_on().await.unwrap();

    // Should not error
    gpio.play_startup_sound().await.unwrap();
    gpio.play_radiation_warning().await.unwrap();
}

#[tokio::test]
async fn test_gpio_beam_stop_control() {
    let gpio = TestGpio::new();
    gpio.power_on().await.unwrap();

    // Should be able to open/close beam stop
    gpio.open_beam_stop().await.unwrap();
    gpio.close_beam_stop().await.unwrap();
}

// ============================================================================
// DemoGpio Specific Tests
// ============================================================================

#[tokio::test]
async fn test_demo_gpio_interlock_simulation() {
    let gpio = DemoGpio::new();
    gpio.power_on().await.unwrap();

    // Initially all interlocks are safe
    assert!(gpio.is_safe_to_operate().await);

    // Trigger emergency stop
    gpio.set_emergency_stop(true).await.unwrap();
    
    // Wait for watchdog to update
    tokio::time::sleep(tokio::time::Duration::from_millis(150)).await;
    
    let interlocks = gpio.get_interlocks().await;
    assert!(!interlocks.emergency_stop);
    assert!(!interlocks.overall_safe);
    assert!(!gpio.is_safe_to_operate().await);

    // Release emergency stop
    gpio.set_emergency_stop(false).await.unwrap();
    
    // Wait for watchdog to update
    tokio::time::sleep(tokio::time::Duration::from_millis(150)).await;
    
    assert!(gpio.is_safe_to_operate().await);
}

#[tokio::test]
async fn test_demo_gpio_door_interlock() {
    let gpio = DemoGpio::new();
    gpio.power_on().await.unwrap();

    // Open door
    gpio.set_door_closed(false).await.unwrap();
    
    // Wait for watchdog
    tokio::time::sleep(tokio::time::Duration::from_millis(150)).await;
    
    let interlocks = gpio.get_interlocks().await;
    assert!(!interlocks.door_closed);
    assert!(!interlocks.overall_safe);

    // Close door
    gpio.set_door_closed(true).await.unwrap();
    
    // Wait for watchdog
    tokio::time::sleep(tokio::time::Duration::from_millis(150)).await;
    
    assert!(gpio.is_safe_to_operate().await);
}

#[tokio::test]
async fn test_demo_gpio_cooling_and_power_interlocks() {
    let gpio = DemoGpio::new();
    gpio.power_on().await.unwrap();

    // Fail cooling
    gpio.set_cooling_ok(false).await.unwrap();
    tokio::time::sleep(tokio::time::Duration::from_millis(150)).await;
    
    let interlocks = gpio.get_interlocks().await;
    assert!(!interlocks.cooling_ok);
    assert!(!interlocks.overall_safe);

    // Restore cooling
    gpio.set_cooling_ok(true).await.unwrap();
    tokio::time::sleep(tokio::time::Duration::from_millis(150)).await;

    // Fail power
    gpio.set_power_ok(false).await.unwrap();
    tokio::time::sleep(tokio::time::Duration::from_millis(150)).await;
    
    let interlocks = gpio.get_interlocks().await;
    assert!(!interlocks.power_ok);
    assert!(!interlocks.overall_safe);

    // Restore power
    gpio.set_power_ok(true).await.unwrap();
    tokio::time::sleep(tokio::time::Duration::from_millis(150)).await;
    
    assert!(gpio.is_safe_to_operate().await);
}

#[tokio::test]
async fn test_demo_gpio_beam_stop_watchdog() {
    let gpio = DemoGpio::new();
    gpio.power_on().await.unwrap();

    // Initially radiation should be safe (beam-stop closed)
    let interlocks = gpio.get_interlocks().await;
    assert!(interlocks.radiation_safe);

    // Open beam-stop
    gpio.open_beam_stop().await.unwrap();

    // Wait for mechanical delay (simulated ~200ms)
    tokio::time::sleep(tokio::time::Duration::from_millis(250)).await;

    // Radiation should now be unsafe
    let interlocks = gpio.get_interlocks().await;
    assert!(!interlocks.radiation_safe);
    assert!(!interlocks.overall_safe);

    // Close beam-stop
    gpio.close_beam_stop().await.unwrap();

    // Wait for mechanical delay (simulated ~150ms)
    tokio::time::sleep(tokio::time::Duration::from_millis(200)).await;

    // Radiation should be safe again
    let interlocks = gpio.get_interlocks().await;
    assert!(interlocks.radiation_safe);
    assert!(interlocks.overall_safe);
}

#[tokio::test]
async fn test_demo_gpio_activation_button_timeout() {
    let gpio = DemoGpio::with_config(2); // 2 second timeout
    gpio.power_on().await.unwrap();

    // Initially not active
    assert!(!gpio.get_activation_button_active().await.unwrap());

    // Activate
    gpio.activate_enable_button().await.unwrap();

    // Should be active
    assert!(gpio.get_activation_button_active().await.unwrap());

    // Should have remaining time
    let remaining = gpio.get_activation_button_remaining_time().await;
    assert!(remaining.is_some());
    assert!(remaining.unwrap() > 0);

    // Wait for timeout
    tokio::time::sleep(tokio::time::Duration::from_secs(3)).await;

    // Should be inactive now
    assert!(!gpio.get_activation_button_active().await.unwrap());
    assert!(gpio.get_activation_button_remaining_time().await.is_none());
}

#[tokio::test]
async fn test_demo_gpio_key_switch() {
    let gpio = DemoGpio::new();
    gpio.power_on().await.unwrap();

    // Initially OFF
    assert!(!gpio.get_key_switch_state().await.unwrap());

    // Turn ON
    gpio.set_key_switch(true).await.unwrap();
    
    // Wait for watchdog
    tokio::time::sleep(tokio::time::Duration::from_millis(150)).await;
    
    assert!(gpio.get_key_switch_state().await.unwrap());

    // Turn OFF
    gpio.set_key_switch(false).await.unwrap();
    
    // Wait for watchdog
    tokio::time::sleep(tokio::time::Duration::from_millis(150)).await;
    
    assert!(!gpio.get_key_switch_state().await.unwrap());
}

#[tokio::test]
async fn test_demo_gpio_arm_all_interlocks() {
    let gpio = DemoGpio::new();
    gpio.power_on().await.unwrap();

    // Trigger multiple faults
    gpio.set_emergency_stop(true).await.unwrap();
    gpio.set_door_closed(false).await.unwrap();
    gpio.set_cooling_ok(false).await.unwrap();
    
    // Wait for watchdog
    tokio::time::sleep(tokio::time::Duration::from_millis(150)).await;

    // Should be unsafe
    assert!(!gpio.is_safe_to_operate().await);

    // Arm all interlocks
    gpio.arm_all_interlocks().await.unwrap();
    
    // Wait for watchdog
    tokio::time::sleep(tokio::time::Duration::from_millis(150)).await;

    // Should be safe now
    assert!(gpio.is_safe_to_operate().await);
    
    let interlocks = gpio.get_interlocks().await;
    assert!(interlocks.emergency_stop);
    assert!(interlocks.door_closed);
    assert!(interlocks.radiation_safe);
    assert!(interlocks.cooling_ok);
    assert!(interlocks.power_ok);
    assert!(interlocks.overall_safe);
}

#[tokio::test]
async fn test_demo_gpio_power_off_clears_outputs() {
    let gpio = DemoGpio::new();
    gpio.power_on().await.unwrap();

    // Set some output pins
    let pins = gpio.get_pins().await;
    assert!(!pins.is_empty());

    // Power off
    gpio.power_off().await.unwrap();

    // All output pins should be low
    let pins = gpio.get_pins().await;
    for pin in pins.iter() {
        if pin.mode == GpioPinMode::Output {
            assert_eq!(pin.state, GpioPinState::Low);
        }
    }
}

#[tokio::test]
async fn test_demo_gpio_pre_configured_pins() {
    let gpio = DemoGpio::new();
    gpio.power_on().await.unwrap();

    // DemoGpio pre-configures interlock pins
    let pins = gpio.get_pins().await;
    assert!(pins.len() >= 5, "Should have at least 5 interlock pins configured");

    // Find emergency stop pin (pin 1)
    let e_stop = pins.iter().find(|p| p.pin == 1);
    assert!(e_stop.is_some());
    assert_eq!(e_stop.unwrap().mode, GpioPinMode::Input);
    assert_eq!(e_stop.unwrap().name, Some("Emergency Stop".to_string()));

    // Find door closed pin (pin 2)
    let door = pins.iter().find(|p| p.pin == 2);
    assert!(door.is_some());
    assert_eq!(door.unwrap().mode, GpioPinMode::Input);
}
