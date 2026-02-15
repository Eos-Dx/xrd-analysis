use aes_gcm::{
    aead::{Aead, KeyInit, OsRng},
    Aes256Gcm, Key,
};
use anyhow::{Result, anyhow};
use base64::{Engine as _, engine::general_purpose};
use getrandom::getrandom;
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::Path;
use tracing::{info, warn, error};

use super::device_config::DeviceConfig;

const CONFIG_FILE_NAME: &str = "device_config.encrypted";
const KEY_FILE_NAME: &str = "config.key";
const NONCE_SIZE: usize = 12; // 96 bits for GCM

/// Encrypted configuration manager
pub struct ConfigManager {
    config_dir: String,
    key: Option<Key<Aes256Gcm>>,
}

#[derive(Serialize, Deserialize)]
struct EncryptedConfig {
    version: u32,
    nonce: String,     // Base64 encoded nonce
    data: String,      // Base64 encoded encrypted data
    checksum: String,  // Simple integrity check
}

impl ConfigManager {
    pub fn new(config_dir: &str) -> Self {
        Self {
            config_dir: config_dir.to_string(),
            key: None,
        }
    }

    /// Initialize the configuration system - generate key if needed
    pub async fn initialize(&mut self) -> Result<()> {
        // Ensure config directory exists
        fs::create_dir_all(&self.config_dir)?;
        
        let key_path = Path::new(&self.config_dir).join(KEY_FILE_NAME);
        
        if key_path.exists() {
            // Load existing key
            self.load_key(&key_path)?;
            info!("Configuration encryption key loaded");
        } else {
            // Generate new key
            self.generate_and_save_key(&key_path)?;
            info!("New configuration encryption key generated");
        }

        Ok(())
    }

    /// Load configuration from encrypted file, or create default if doesn't exist
    pub async fn load_config(&self) -> Result<DeviceConfig> {
        let config_path = Path::new(&self.config_dir).join(CONFIG_FILE_NAME);
        
        if config_path.exists() {
            match self.decrypt_config(&config_path) {
                Ok(config) => {
                    info!("Device configuration loaded from encrypted file");
                    Ok(config)
                }
                Err(e) => {
                    error!("Failed to decrypt configuration: {}", e);
                    warn!("Using default configuration");
                    Ok(DeviceConfig::default())
                }
            }
        } else {
            info!("No encrypted configuration found, using defaults");
            let default_config = DeviceConfig::default();
            
            // Save default configuration
            if let Err(e) = self.save_config(&default_config).await {
                warn!("Failed to save default configuration: {}", e);
            }
            
            Ok(default_config)
        }
    }

    /// Save configuration to encrypted file
    pub async fn save_config(&self, config: &DeviceConfig) -> Result<()> {
        let config_path = Path::new(&self.config_dir).join(CONFIG_FILE_NAME);
        
        if self.key.is_none() {
            return Err(anyhow!("Encryption key not initialized"));
        }

        let encrypted_config = self.encrypt_config(config)?;
        let json_data = serde_json::to_string_pretty(&encrypted_config)?;
        
        fs::write(&config_path, json_data)?;
        info!("Device configuration saved to encrypted file");
        
        Ok(())
    }

    /// Update configuration in maintenance mode
    pub async fn update_config_maintenance(
        &self, 
        mut config: DeviceConfig, 
        maintenance_password: &str
    ) -> Result<DeviceConfig> {
        // Verify maintenance mode access
        if !config.maintenance.maintenance_mode {
            return Err(anyhow!("Device not in maintenance mode"));
        }

        if let Some(stored_hash) = &config.maintenance.maintenance_password_hash {
            let provided_hash = self.hash_password(maintenance_password);
            if provided_hash != *stored_hash {
                return Err(anyhow!("Invalid maintenance password"));
            }
        }

        // Update configuration metadata
        config.maintenance.last_config_update = Some(chrono::Utc::now());
        config.maintenance.config_version += 1;

        // Save updated configuration
        self.save_config(&config).await?;
        
        info!("Configuration updated in maintenance mode (version {})", config.maintenance.config_version);
        Ok(config)
    }

    /// Enable maintenance mode with password
    pub async fn enable_maintenance_mode(&self, mut config: DeviceConfig, password: &str) -> Result<DeviceConfig> {
        let password_hash = self.hash_password(password);
        
        config.maintenance.maintenance_mode = true;
        config.maintenance.maintenance_password_hash = Some(password_hash);
        
        self.save_config(&config).await?;
        
        warn!("Maintenance mode ENABLED");
        Ok(config)
    }

    /// Disable maintenance mode
    pub async fn disable_maintenance_mode(&self, mut config: DeviceConfig) -> Result<DeviceConfig> {
        config.maintenance.maintenance_mode = false;
        config.maintenance.bypass_interlocks = false;
        config.maintenance.maintenance_password_hash = None;
        
        self.save_config(&config).await?;
        
        info!("Maintenance mode disabled");
        Ok(config)
    }

    /// Generate and save encryption key
    fn generate_and_save_key(&mut self, key_path: &Path) -> Result<()> {
        let key = Aes256Gcm::generate_key(&mut OsRng);
        self.key = Some(key);
        
        // Encode key as base64 for storage
        let key_b64 = general_purpose::STANDARD.encode(&key);
        fs::write(key_path, key_b64)?;
        
        // Set restrictive permissions on Windows (best effort)
        #[cfg(windows)]
        {
            
            // On Windows, we can't easily set Unix-style permissions,
            // but the file is in a protected directory
        }
        
        Ok(())
    }

    /// Load encryption key from file
    fn load_key(&mut self, key_path: &Path) -> Result<()> {
        let key_b64 = fs::read_to_string(key_path)?;
        let key_bytes = general_purpose::STANDARD.decode(key_b64.trim())?;
        
        let key_array: [u8; 32] = key_bytes.try_into()
            .map_err(|_| anyhow!("Invalid key length, expected 32 bytes"))?;
        
        self.key = Some(key_array.into());
        Ok(())
    }

    /// Encrypt configuration data
    fn encrypt_config(&self, config: &DeviceConfig) -> Result<EncryptedConfig> {
        let key = self.key.as_ref().ok_or_else(|| anyhow!("No encryption key"))?;
        let cipher = Aes256Gcm::new(key);
        
        // Serialize config to JSON
        let plaintext = serde_json::to_string(config)?;
        
        // Generate random nonce
        let mut nonce_bytes = [0u8; NONCE_SIZE];
        getrandom(&mut nonce_bytes)?;
        let nonce = &nonce_bytes.into();
        
        // Encrypt
        let ciphertext = cipher.encrypt(nonce, plaintext.as_bytes())
            .map_err(|e| anyhow!("Encryption failed: {}", e))?;
        
        // Calculate simple checksum
        let checksum = self.calculate_checksum(&plaintext);
        
        Ok(EncryptedConfig {
            version: 1,
            nonce: general_purpose::STANDARD.encode(nonce_bytes),
            data: general_purpose::STANDARD.encode(ciphertext),
            checksum,
        })
    }

    /// Decrypt configuration data
    fn decrypt_config(&self, config_path: &Path) -> Result<DeviceConfig> {
        let key = self.key.as_ref().ok_or_else(|| anyhow!("No encryption key"))?;
        let cipher = Aes256Gcm::new(key);
        
        // Load encrypted config
        let json_data = fs::read_to_string(config_path)?;
        let encrypted_config: EncryptedConfig = serde_json::from_str(&json_data)?;
        
        // Decode nonce and data
        let nonce_bytes = general_purpose::STANDARD.decode(&encrypted_config.nonce)?;
        let ciphertext = general_purpose::STANDARD.decode(&encrypted_config.data)?;
        
        let nonce_array: [u8; NONCE_SIZE] = nonce_bytes.try_into()
            .map_err(|_| anyhow!("Invalid nonce size, expected {} bytes", NONCE_SIZE))?;
        
        let nonce = &nonce_array.into();
        
        // Decrypt
        let plaintext_bytes = cipher.decrypt(nonce, ciphertext.as_ref())
            .map_err(|e| anyhow!("Decryption failed: {}", e))?;
        
        let plaintext = String::from_utf8(plaintext_bytes)?;
        
        // Verify checksum
        let calculated_checksum = self.calculate_checksum(&plaintext);
        if calculated_checksum != encrypted_config.checksum {
            return Err(anyhow!("Configuration checksum mismatch"));
        }
        
        // Deserialize config
        let config: DeviceConfig = serde_json::from_str(&plaintext)?;
        Ok(config)
    }

    /// Calculate simple checksum for integrity verification
    fn calculate_checksum(&self, data: &str) -> String {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        
        let mut hasher = DefaultHasher::new();
        data.hash(&mut hasher);
        format!("{:x}", hasher.finish())
    }

    /// Hash password for storage (simple implementation)
    fn hash_password(&self, password: &str) -> String {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        
        let mut hasher = DefaultHasher::new();
        password.hash(&mut hasher);
        format!("{:x}", hasher.finish())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[tokio::test]
    async fn test_config_encryption_roundtrip() {
        let temp_dir = TempDir::new().unwrap();
        let temp_path = temp_dir.path().to_str().unwrap();
        
        let mut manager = ConfigManager::new(temp_path);
        manager.initialize().await.unwrap();
        
        let original_config = DeviceConfig::default();
        manager.save_config(&original_config).await.unwrap();
        
        let loaded_config = manager.load_config().await.unwrap();
        
        // Compare serialized versions since DeviceConfig doesn't implement PartialEq
        let original_json = serde_json::to_string(&original_config).unwrap();
        let loaded_json = serde_json::to_string(&loaded_config).unwrap();
        assert_eq!(original_json, loaded_json);
    }

    #[tokio::test]
    async fn test_maintenance_mode() {
        let temp_dir = TempDir::new().unwrap();
        let temp_path = temp_dir.path().to_str().unwrap();
        
        let mut manager = ConfigManager::new(temp_path);
        manager.initialize().await.unwrap();
        
        let config = DeviceConfig::default();
        let password = "maintenance123";
        
        let config = manager.enable_maintenance_mode(config, password).await.unwrap();
        assert!(config.maintenance.maintenance_mode);
        
        let config = manager.disable_maintenance_mode(config).await.unwrap();
        assert!(!config.maintenance.maintenance_mode);
    }
}