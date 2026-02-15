use anyhow::{Result, Context, anyhow};
use rustls::ServerConfig;
use rustls::server::WebPkiClientVerifier;
use rustls::pki_types::{CertificateDer, PrivateKeyDer};
use std::fs::File;
use std::io::BufReader;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use tracing::{info, warn};
use x509_parser::prelude::*;

/// Certificate configuration for the server
#[derive(Debug, Clone)]
pub struct CertificateConfig {
    /// Server certificate path (e.g., certs/server/server_ABC123.crt)
    pub server_cert_path: PathBuf,
    
    /// Server private key path (e.g., certs/server/server_ABC123.key)
    pub server_key_path: PathBuf,
    
    /// Client Root CA certificate path (e.g., certs/root/maintenance_client_root_ca.crt)
    pub client_ca_cert_path: PathBuf,
    
    /// Device UUID extracted from certificate
    pub device_uuid: Option<String>,
}

impl CertificateConfig {
    /// Create a new certificate configuration
    pub fn new(
        server_cert_path: impl Into<PathBuf>,
        server_key_path: impl Into<PathBuf>,
        client_ca_cert_path: impl Into<PathBuf>,
    ) -> Self {
        Self {
            server_cert_path: server_cert_path.into(),
            server_key_path: server_key_path.into(),
            client_ca_cert_path: client_ca_cert_path.into(),
            device_uuid: None,
        }
    }
    
    /// Auto-detect certificate paths for a given device UUID
    pub fn from_device_uuid(device_uuid: &str, cert_base_dir: impl AsRef<Path>) -> Self {
        let base = cert_base_dir.as_ref();
        Self {
            server_cert_path: base.join("server").join(format!("server_{}.crt", device_uuid)),
            server_key_path: base.join("server").join(format!("server_{}.key", device_uuid)),
            client_ca_cert_path: base.join("root").join("maintenance_client_root_ca.crt"),
            device_uuid: Some(device_uuid.to_string()),
        }
    }
}

/// Certificate loader for mTLS configuration
pub struct CertificateLoader {
    config: CertificateConfig,
}

impl CertificateLoader {
    /// Create a new certificate loader
    pub fn new(config: CertificateConfig) -> Self {
        Self { config }
    }
    
    /// Load certificates and create rustls ServerConfig for mTLS
    pub fn load_server_config(&self) -> Result<ServerConfig> {
        info!("🔐 Loading server certificates for mTLS...");
        
        // Load server certificate
        let server_certs = self.load_certificates(&self.config.server_cert_path)
            .context("Failed to load server certificate")?;
        
        if server_certs.is_empty() {
            return Err(anyhow!("Server certificate file is empty"));
        }
        
        info!("  ✓ Loaded server certificate: {}", self.config.server_cert_path.display());
        
        // Validate and extract device UUID from server certificate
        if let Some(expected_uuid) = &self.config.device_uuid {
            let extracted_uuid = self.extract_device_uuid(&server_certs[0])?;
            if extracted_uuid != *expected_uuid {
                warn!("⚠️  Device UUID mismatch: expected '{}', found '{}'", expected_uuid, extracted_uuid);
            } else {
                info!("  ✓ Device UUID validated: {}", extracted_uuid);
            }
        }
        
        // Load server private key
        let server_key = self.load_private_key(&self.config.server_key_path)
            .context("Failed to load server private key")?;
        
        info!("  ✓ Loaded server private key: {}", self.config.server_key_path.display());
        
        // Load client CA certificate for client verification
        let client_ca_certs = self.load_certificates(&self.config.client_ca_cert_path)
            .context("Failed to load client CA certificate")?;
        
        if client_ca_certs.is_empty() {
            return Err(anyhow!("Client CA certificate file is empty"));
        }
        
        info!("  ✓ Loaded client CA certificate: {}", self.config.client_ca_cert_path.display());
        
        // Create client verifier that requires and validates client certificates
        let mut root_store = rustls::RootCertStore::empty();
        for cert in client_ca_certs {
            root_store.add(cert).context("Failed to add client CA to root store")?;
        }
        
        let client_verifier = WebPkiClientVerifier::builder(Arc::new(root_store))
            .build()
            .context("Failed to build client certificate verifier")?;
        
        // Build server config with mutual TLS
        let server_config = ServerConfig::builder()
            .with_client_cert_verifier(client_verifier)
            .with_single_cert(server_certs, server_key)
            .context("Failed to configure server with certificates")?;
        
        info!("🔒 mTLS server configuration complete");
        info!("  - Server authentication: ENABLED");
        info!("  - Client authentication: REQUIRED");
        info!("  - Certificate validation: ENFORCED");
        
        Ok(server_config)
    }
    
    /// Load certificates from a PEM file
    fn load_certificates(&self, path: &Path) -> Result<Vec<CertificateDer<'static>>> {
        let file = File::open(path)
            .with_context(|| format!("Failed to open certificate file: {}", path.display()))?;
        
        let mut reader = BufReader::new(file);
        let certs = rustls_pemfile::certs(&mut reader)
            .collect::<Result<Vec<_>, _>>()
            .with_context(|| format!("Failed to parse certificates from: {}", path.display()))?;
        
        if certs.is_empty() {
            return Err(anyhow!("No certificates found in file: {}", path.display()));
        }
        
        Ok(certs)
    }
    
    /// Load private key from a PEM file
    fn load_private_key(&self, path: &Path) -> Result<PrivateKeyDer<'static>> {
        let file = File::open(path)
            .with_context(|| format!("Failed to open private key file: {}", path.display()))?;
        
        let mut reader = BufReader::new(file);
        
        // Try PKCS8 format first
        if let Some(key) = rustls_pemfile::pkcs8_private_keys(&mut reader)
            .next()
            .transpose()
            .with_context(|| format!("Failed to parse PKCS8 key from: {}", path.display()))? 
        {
            return Ok(PrivateKeyDer::Pkcs8(key));
        }
        
        // Reset reader and try RSA format
        let file = File::open(path)?;
        let mut reader = BufReader::new(file);
        
        if let Some(key) = rustls_pemfile::rsa_private_keys(&mut reader)
            .next()
            .transpose()
            .with_context(|| format!("Failed to parse RSA key from: {}", path.display()))? 
        {
            return Ok(PrivateKeyDer::Pkcs1(key));
        }
        
        Err(anyhow!("No valid private key found in file: {}", path.display()))
    }
    
    /// Extract device UUID from server certificate SAN (Subject Alternative Name)
    fn extract_device_uuid(&self, cert_der: &CertificateDer) -> Result<String> {
        let (_, cert) = X509Certificate::from_der(cert_der)
            .map_err(|e| anyhow!("Failed to parse X.509 certificate: {}", e))?;
        
        // Look for SAN extension
        if let Some(san_ext) = cert.subject_alternative_name()
            .map_err(|e| anyhow!("Failed to read SAN extension: {}", e))? 
        {
            for general_name in &san_ext.value.general_names {
                if let x509_parser::extensions::GeneralName::URI(uri) = general_name {
                    // Parse URI like "urn:omniscan:server:ABC123"
                    if uri.starts_with("urn:omniscan:server:") {
                        let uuid = uri.strip_prefix("urn:omniscan:server:").unwrap_or("");
                        if !uuid.is_empty() {
                            return Ok(uuid.to_string());
                        }
                    }
                }
            }
        }
        
        Err(anyhow!("Device UUID not found in certificate SAN"))
    }
}

/// Extract client information from a peer certificate
pub struct ClientCertInfo {
    pub engineer_id: String,
    pub device_uuid: String,
    pub common_name: String,
}

impl ClientCertInfo {
    /// Parse client certificate information from DER-encoded certificate
    pub fn from_der(cert_der: &[u8]) -> Result<Self> {
        let (_, cert) = X509Certificate::from_der(cert_der)
            .map_err(|e| anyhow!("Failed to parse client certificate: {}", e))?;
        
        // Extract CN from subject
        let common_name = cert.subject()
            .iter_common_name()
            .next()
            .and_then(|cn| cn.as_str().ok())
            .ok_or_else(|| anyhow!("Common Name not found in client certificate"))?
            .to_string();
        
        // Extract engineer ID from CN (e.g., "Maintenance Engineer ENG001")
        let engineer_id = common_name
            .strip_prefix("Maintenance Engineer ")
            .unwrap_or(&common_name)
            .to_string();
        
        // Extract device UUID from SAN
        let mut device_uuid = String::new();
        if let Some(san_ext) = cert.subject_alternative_name()
            .map_err(|e| anyhow!("Failed to read SAN extension: {}", e))? 
        {
            for general_name in &san_ext.value.general_names {
                if let x509_parser::extensions::GeneralName::URI(uri) = general_name {
                    // Parse URI like "urn:omniscan:server:ABC123"
                    if uri.starts_with("urn:omniscan:server:") {
                        device_uuid = uri.strip_prefix("urn:omniscan:server:")
                            .unwrap_or("")
                            .to_string();
                        break;
                    }
                }
            }
        }
        
        if device_uuid.is_empty() {
            return Err(anyhow!("Device UUID not found in client certificate SAN"));
        }
        
        Ok(Self {
            engineer_id,
            device_uuid,
            common_name,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_certificate_config_from_uuid() {
        let config = CertificateConfig::from_device_uuid("ABC123", "certs");
        
        assert_eq!(config.server_cert_path, PathBuf::from("certs/server/server_ABC123.crt"));
        assert_eq!(config.server_key_path, PathBuf::from("certs/server/server_ABC123.key"));
        assert_eq!(config.client_ca_cert_path, PathBuf::from("certs/root/maintenance_client_root_ca.crt"));
        assert_eq!(config.device_uuid, Some("ABC123".to_string()));
    }
}
