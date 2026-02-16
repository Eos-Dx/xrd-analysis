pub mod device_config;
pub mod encryption;

use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::path::Path;

pub use device_config::*;
pub use encryption::ConfigManager;

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct DeviceConfig {
    pub name: String,
    pub interlocks_armed: bool,
    pub emergency_stop: bool,
    pub power_default_on: bool,
    #[serde(default)]
    pub gui_mode: bool,  // Enable GUI control panel for DEMO mode
}

impl Default for DeviceConfig {
    fn default() -> Self {
        Self {
            name: "GPIO".to_string(),
            interlocks_armed: true,
            emergency_stop: false,
            power_default_on: true,
            gui_mode: false,  // Default to no GUI (production mode)
        }
    }
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct MeasurementConfig {
    pub output_dir: String,
    pub default_duration_s: u32,
}

impl Default for MeasurementConfig {
    fn default() -> Self {
        Self {
            output_dir: "measurements".to_string(),
            default_duration_s: 60,
        }
    }
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct LoggingConfig {
    pub log_dir: String,
    pub log_level: String,
}

impl Default for LoggingConfig {
    fn default() -> Self {
        Self {
            log_dir: "logs".to_string(),
            log_level: "INFO".to_string(),
        }
    }
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct MaintenanceConfig {
    pub dev_mode: bool,
    pub dev_secret: String,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct WorkflowConfig {
    /// Enable button timeout (seconds)
    #[serde(default = "default_enable_button_timeout")]
    pub enable_button_timeout: u32,
    
    /// Warmup duration (seconds)
    #[serde(default = "default_warmup_duration")]
    pub warmup_duration: u32,
    
    /// Calibration validity (hours)
    #[serde(default = "default_calibration_validity_hours")]
    pub calibration_validity_hours: i64,
    
    /// GPIO polling interval (milliseconds)
    #[serde(default = "default_gpio_poll_interval_ms")]
    pub gpio_poll_interval_ms: u64,
}

fn default_enable_button_timeout() -> u32 { 20 }
fn default_warmup_duration() -> u32 { 600 }
fn default_calibration_validity_hours() -> i64 { 24 }
fn default_gpio_poll_interval_ms() -> u64 { 100 }

impl Default for WorkflowConfig {
    fn default() -> Self {
        Self {
            enable_button_timeout: 20,
            warmup_duration: 600,
            calibration_validity_hours: 24,
            gpio_poll_interval_ms: 100,
        }
    }
}

impl Default for MaintenanceConfig {
    fn default() -> Self {
        Self {
            dev_mode: true,
            dev_secret: "change-me".to_string(),
        }
    }
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct CertificateConfig {
    /// Device UUID (e.g., "ABC123")
    pub device_uuid: String,
    
    /// Device Certificate Number (for audit trail tracking)
    /// This identifies which specific machine performed measurements
    pub device_certificate_number: String,
    
    /// Base directory for certificates (e.g., "../omniscan-certificate-center/certs")
    pub cert_base_dir: String,
    
    /// Enable mTLS (if false, run in plaintext mode for development)
    #[serde(default = "default_mtls_enabled")]
    pub enable_mtls: bool,
}

fn default_mtls_enabled() -> bool {
    true
}

impl Default for CertificateConfig {
    fn default() -> Self {
        Self {
            device_uuid: "DEMO-001".to_string(),
            device_certificate_number: "CERT-DEMO-001".to_string(),
            cert_base_dir: "../omniscan-certificate-center/certs".to_string(),
            enable_mtls: false, // Default to false for development
        }
    }
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ServerConfig {
    pub version: u32,
    #[serde(default)]
    pub device: DeviceConfig,
    #[serde(default)]
    pub measurement: MeasurementConfig,
    #[serde(default)]
    pub logging: LoggingConfig,
    #[serde(default)]
    pub maintenance: MaintenanceConfig,
    #[serde(default)]
    pub certificates: CertificateConfig,
    #[serde(default)]
    pub workflow: WorkflowConfig,
}

impl Default for ServerConfig {
    fn default() -> Self {
        Self {
            version: 1,
            device: DeviceConfig::default(),
            measurement: MeasurementConfig::default(),
            logging: LoggingConfig::default(),
            maintenance: MaintenanceConfig::default(),
            certificates: CertificateConfig::default(),
            workflow: WorkflowConfig::default(),
        }
    }
}

impl ServerConfig {
    pub fn load_from_file<P: AsRef<Path>>(path: P) -> Result<Self> {
        let path = path.as_ref();
        let content = std::fs::read_to_string(path)?;
        
        // Detect format by file extension
        let config: ServerConfig = if path.extension().and_then(|s| s.to_str()) == Some("toml") {
            toml::from_str(&content)?
        } else {
            serde_json::from_str(&content)?
        };
        
        // Ensure directories exist
        std::fs::create_dir_all(&config.logging.log_dir)?;
        std::fs::create_dir_all(&config.measurement.output_dir)?;
        
        Ok(config)
    }
    
    /// Save configuration back to disk (supports .toml or .json based on path extension)
    pub fn save_to_file<P: AsRef<Path>>(&self, path: P) -> Result<()> {
        let path = path.as_ref();
        let serialized = if path.extension().and_then(|s| s.to_str()) == Some("toml") {
            toml::to_string_pretty(self)?
        } else {
            serde_json::to_string_pretty(self)?
        };
        std::fs::write(path, serialized)?;
        Ok(())
    }
    
    pub fn validate(&self) -> Result<()> {
        if self.version == 0 {
            return Err(anyhow::anyhow!("Invalid config version"));
        }
        
        if self.device.name.is_empty() {
            return Err(anyhow::anyhow!("Device name cannot be empty"));
        }
        
        if self.measurement.default_duration_s == 0 {
            return Err(anyhow::anyhow!("Default measurement duration must be > 0"));
        }
        
        Ok(())
    }
}
