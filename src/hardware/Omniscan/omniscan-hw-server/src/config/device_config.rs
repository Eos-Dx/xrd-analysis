use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Motion controller configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MotionConfig {
    pub name: String,
    pub enabled: bool,
    pub x_axis: AxisConfig,
    pub y_axis: AxisConfig,
    pub homing_speed: f64,         // mm/s
    pub max_speed: f64,            // mm/s  
    pub acceleration: f64,         // mm/s²
    pub backlash_compensation: f64, // mm
}

impl Default for MotionConfig {
    fn default() -> Self {
        Self {
            name: "Default Motion Controller".to_string(),
            enabled: true,
            x_axis: AxisConfig::default(),
            y_axis: AxisConfig::default(),
            homing_speed: 10.0,
            max_speed: 50.0,
            acceleration: 100.0,
            backlash_compensation: 0.01,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AxisConfig {
    pub enabled: bool,
    pub min_position: f64,         // mm
    pub max_position: f64,         // mm
    pub home_position: f64,        // mm
    pub steps_per_mm: f64,
    pub max_velocity: f64,         // mm/s
    pub acceleration: f64,         // mm/s²
}

impl Default for AxisConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            min_position: 0.0,
            max_position: 100.0,
            home_position: 0.0,
            steps_per_mm: 1000.0,
            max_velocity: 50.0,
            acceleration: 100.0,
        }
    }
}

/// Detector configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DetectorConfig {
    pub name: String,
    pub detector_type: String,     // "demo", "bruker_bis", "advocam"
    pub enabled: bool,
    pub max_exposure_time: u32,    // milliseconds
    pub min_exposure_time: u32,    // milliseconds
    pub temperature_limits: TemperatureLimits,
    pub voltage_limits: VoltageLimits,
    pub data_path: String,
    pub calibration_interval: u32, // hours
    // Type-specific settings
    pub specific_settings: HashMap<String, serde_json::Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TemperatureLimits {
    pub min_operating: f32,        // °C
    pub max_operating: f32,        // °C
    pub warning_threshold: f32,    // °C
    pub shutdown_threshold: f32,   // °C
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VoltageLimits {
    pub nominal: f32,              // V
    pub tolerance: f32,            // V (±)
    pub min_operational: f32,      // V
    pub max_operational: f32,      // V
}

/// GPIO configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpioConfig {
    pub name: String,
    pub enabled: bool,
    pub pin_mappings: HashMap<u8, PinConfig>,
    pub interlock_config: InterlockConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PinConfig {
    pub name: String,
    pub pin_type: String,          // "input", "output", "input_pullup", "input_pulldown"
    pub function: String,          // "emergency_stop", "door_sensor", "ready_led", etc.
    pub active_state: String,      // "high", "low"
    pub debounce_ms: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InterlockConfig {
    pub emergency_stop_pin: u8,
    pub door_sensor_pin: u8,
    pub radiation_sensor_pin: u8,
    pub cooling_sensor_pin: u8,
    pub power_monitor_pin: u8,
    pub ready_led_pin: u8,
    pub warning_led_pin: u8,
    pub bypass_mode: bool,         // For maintenance only
    pub required_interlocks: Vec<String>,
}

/// Safety configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SafetyConfig {
    pub max_exposure_time: u32,    // milliseconds
    pub max_daily_exposures: u32,
    pub enable_physical_confirmation: bool,
    pub arm_window_timeout: u32,   // seconds
    pub watchdog_timeout: u32,     // milliseconds
    pub emergency_procedures: EmergencyConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmergencyConfig {
    pub auto_power_down: bool,
    pub auto_home_motors: bool,
    pub send_notifications: bool,
    pub log_level: String,         // "error", "warn", "info", "debug"
}

/// Complete device configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeviceConfig {
    pub device_info: DeviceInfo,
    pub motion: MotionConfig,
    pub detector: DetectorConfig,
    pub gpio: GpioConfig,
    pub safety: SafetyConfig,
    pub maintenance: MaintenanceConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeviceInfo {
    pub device_id: String,
    pub device_name: String,
    pub serial_number: String,
    pub firmware_version: String,
    pub calibration_date: Option<chrono::DateTime<chrono::Utc>>,
    pub last_service_date: Option<chrono::DateTime<chrono::Utc>>,
    pub location: String,
    pub operator: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MaintenanceConfig {
    pub maintenance_mode: bool,
    pub bypass_interlocks: bool,
    pub engineer_access_level: u8,
    pub maintenance_password_hash: Option<String>,
    pub last_config_update: Option<chrono::DateTime<chrono::Utc>>,
    pub config_version: u32,
}

impl Default for DeviceConfig {
    fn default() -> Self {
        Self {
            device_info: DeviceInfo {
                device_id: uuid::Uuid::new_v4().to_string(),
                device_name: "Omniscan DEMO Device".to_string(),
                serial_number: "DEMO-001".to_string(),
                firmware_version: env!("CARGO_PKG_VERSION").to_string(),
                calibration_date: None,
                last_service_date: None,
                location: "Laboratory".to_string(),
                operator: None,
            },
            motion: MotionConfig {
                name: "XY Stage Controller".to_string(),
                enabled: true,
                x_axis: AxisConfig {
                    enabled: true,
                    min_position: 0.0,
                    max_position: 100.0,
                    home_position: 0.0,
                    steps_per_mm: 1000.0,
                    max_velocity: 50.0,
                    acceleration: 100.0,
                },
                y_axis: AxisConfig {
                    enabled: true,
                    min_position: 0.0,
                    max_position: 100.0,
                    home_position: 0.0,
                    steps_per_mm: 1000.0,
                    max_velocity: 50.0,
                    acceleration: 100.0,
                },
                homing_speed: 10.0,
                max_speed: 50.0,
                acceleration: 100.0,
                backlash_compensation: 0.01,
            },
            detector: DetectorConfig {
                name: "DEMO Detector".to_string(),
                detector_type: "demo".to_string(),
                enabled: true,
                max_exposure_time: 300_000, // 5 minutes
                min_exposure_time: 1,
                temperature_limits: TemperatureLimits {
                    min_operating: -10.0,
                    max_operating: 60.0,
                    warning_threshold: 50.0,
                    shutdown_threshold: 65.0,
                },
                voltage_limits: VoltageLimits {
                    nominal: 12.0,
                    tolerance: 0.5,
                    min_operational: 11.0,
                    max_operational: 13.0,
                },
                data_path: "./data".to_string(),
                calibration_interval: 168, // 1 week
                specific_settings: HashMap::new(),
            },
            gpio: GpioConfig {
                name: "Safety GPIO Controller".to_string(),
                enabled: true,
                pin_mappings: Self::default_pin_mappings(),
                interlock_config: InterlockConfig {
                    emergency_stop_pin: 1,
                    door_sensor_pin: 2,
                    radiation_sensor_pin: 3,
                    cooling_sensor_pin: 4,
                    power_monitor_pin: 5,
                    ready_led_pin: 10,
                    warning_led_pin: 11,
                    bypass_mode: false,
                    required_interlocks: vec![
                        "emergency_stop".to_string(),
                        "door_closed".to_string(),
                        "radiation_safe".to_string(),
                        "cooling_ok".to_string(),
                        "power_ok".to_string(),
                    ],
                },
            },
            safety: SafetyConfig {
                max_exposure_time: 300_000,
                max_daily_exposures: 1000,
                enable_physical_confirmation: true,
                arm_window_timeout: 30,
                watchdog_timeout: 5000,
                emergency_procedures: EmergencyConfig {
                    auto_power_down: true,
                    auto_home_motors: true,
                    send_notifications: true,
                    log_level: "info".to_string(),
                },
            },
            maintenance: MaintenanceConfig {
                maintenance_mode: false,
                bypass_interlocks: false,
                engineer_access_level: 0,
                maintenance_password_hash: None,
                last_config_update: Some(chrono::Utc::now()),
                config_version: 1,
            },
        }
    }
}

impl DeviceConfig {
    fn default_pin_mappings() -> HashMap<u8, PinConfig> {
        let mut pins = HashMap::new();
        
        pins.insert(1, PinConfig {
            name: "Emergency Stop".to_string(),
            pin_type: "input_pullup".to_string(),
            function: "emergency_stop".to_string(),
            active_state: "low".to_string(),
            debounce_ms: 50,
        });
        
        pins.insert(2, PinConfig {
            name: "Door Closed Sensor".to_string(),
            pin_type: "input_pullup".to_string(),
            function: "door_sensor".to_string(),
            active_state: "high".to_string(),
            debounce_ms: 100,
        });
        
        pins.insert(3, PinConfig {
            name: "Radiation Monitor".to_string(),
            pin_type: "input".to_string(),
            function: "radiation_sensor".to_string(),
            active_state: "high".to_string(),
            debounce_ms: 200,
        });
        
        pins.insert(4, PinConfig {
            name: "Cooling Status".to_string(),
            pin_type: "input".to_string(),
            function: "cooling_sensor".to_string(),
            active_state: "high".to_string(),
            debounce_ms: 500,
        });
        
        pins.insert(5, PinConfig {
            name: "Power Monitor".to_string(),
            pin_type: "input".to_string(),
            function: "power_monitor".to_string(),
            active_state: "high".to_string(),
            debounce_ms: 100,
        });
        
        pins.insert(10, PinConfig {
            name: "Ready LED".to_string(),
            pin_type: "output".to_string(),
            function: "ready_led".to_string(),
            active_state: "high".to_string(),
            debounce_ms: 0,
        });
        
        pins.insert(11, PinConfig {
            name: "Warning LED".to_string(),
            pin_type: "output".to_string(),
            function: "warning_led".to_string(),
            active_state: "high".to_string(),
            debounce_ms: 0,
        });
        
        pins
    }
}