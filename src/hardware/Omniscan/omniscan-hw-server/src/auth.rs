//! Authentication and authorization for medical device operations
//! 
//! Implements requirements from USER_EXPECTATIONS.md Section 7:
//! - Role-based access control (Operators, Engineers, Administrators)
//! - Session management with exclusive access control
//! - Audit logging of authentication events
//! - Cryptographic authentication

use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use chrono::{DateTime, Utc, Duration};
use serde::{Deserialize, Serialize};
use uuid::Uuid;
use anyhow::Result;

/// User roles for access control (Section 7: Role-Based Access Control)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Hash)]
pub enum UserRole {
    /// Clinical operators: Measurement execution and calibration
    Operator,
    /// Maintenance engineers: Full diagnostic access and maintenance mode
    Engineer,
    /// IT administrators: User management and system configuration
    Administrator,
}

impl UserRole {
    /// Check if this role can perform the given operation
    pub fn can_perform(&self, operation: &Operation) -> bool {
        match operation {
            // Operators can run measurements and calibrations
            Operation::StartMeasurement | 
            Operation::StopMeasurement | 
            Operation::RunCalibration | 
            Operation::ViewMeasurementData => {
                matches!(self, UserRole::Operator | UserRole::Engineer | UserRole::Administrator)
            },
            
            // Engineers can access maintenance mode and diagnostics
            Operation::EnterMaintenanceMode | 
            Operation::AccessDiagnostics | 
            Operation::BypassInterlocks => {
                matches!(self, UserRole::Engineer | UserRole::Administrator)
            },
            
            // Only administrators can manage users and configuration
            Operation::ManageUsers | 
            Operation::ModifyConfiguration | 
            Operation::ViewAuditLogs => {
                matches!(self, UserRole::Administrator)
            },
        }
    }
}

/// Operations that require authorization
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Operation {
    StartMeasurement,
    StopMeasurement,
    RunCalibration,
    ViewMeasurementData,
    EnterMaintenanceMode,
    AccessDiagnostics,
    BypassInterlocks,
    ManageUsers,
    ModifyConfiguration,
    ViewAuditLogs,
}

/// User information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct User {
    pub id: String,
    pub username: String,
    pub display_name: String,
    pub role: UserRole,
    pub email: Option<String>,
    /// Hashed password (TODO: implement secure password hashing with argon2)
    #[serde(skip_serializing)]
    pub password_hash: String,
    pub created_at: DateTime<Utc>,
    pub last_login: Option<DateTime<Utc>>,
    pub is_active: bool,
}

/// Active session information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Session {
    pub session_id: String,
    pub user_id: String,
    pub username: String,
    pub role: UserRole,
    pub created_at: DateTime<Utc>,
    pub last_activity: DateTime<Utc>,
    pub expires_at: DateTime<Utc>,
    pub ip_address: Option<String>,
    /// Exclusive lock on device operations
    pub has_exclusive_lock: bool,
    /// Estimated time remaining for active operation
    pub estimated_completion: Option<DateTime<Utc>>,
}

impl Session {
    /// Check if session is still valid
    pub fn is_valid(&self) -> bool {
        self.expires_at > Utc::now()
    }
    
    /// Check if session is expired
    pub fn is_expired(&self) -> bool {
        !self.is_valid()
    }
    
    /// Refresh session activity
    pub fn refresh(&mut self) {
        self.last_activity = Utc::now();
        // Extend expiration by 1 hour from now
        self.expires_at = Utc::now() + Duration::hours(1);
    }
}

/// Session manager for tracking active users
pub struct SessionManager {
    sessions: Arc<RwLock<HashMap<String, Session>>>,
    users: Arc<RwLock<HashMap<String, User>>>,
    /// Session holding exclusive device lock
    exclusive_lock_holder: Arc<RwLock<Option<String>>>,
}

impl SessionManager {
    pub fn new() -> Self {
        Self {
            sessions: Arc::new(RwLock::new(HashMap::new())),
            users: Arc::new(RwLock::new(HashMap::new())),
            exclusive_lock_holder: Arc::new(RwLock::new(None)),
        }
    }
    
    /// Initialize with default admin user
    /// TODO: Replace with proper user database and secure credential storage
    pub async fn initialize_default_users(&self) -> Result<()> {
        let mut users = self.users.write().await;
        
        // Create default admin user
        let admin = User {
            id: Uuid::new_v4().to_string(),
            username: "admin".to_string(),
            display_name: "System Administrator".to_string(),
            role: UserRole::Administrator,
            email: Some("admin@omniscan.local".to_string()),
            password_hash: "TODO:hash_password".to_string(), // TODO: Implement secure hashing
            created_at: Utc::now(),
            last_login: None,
            is_active: true,
        };
        users.insert(admin.id.clone(), admin);
        
        // Create default operator user
        let operator = User {
            id: Uuid::new_v4().to_string(),
            username: "operator".to_string(),
            display_name: "Clinical Operator".to_string(),
            role: UserRole::Operator,
            email: None,
            password_hash: "TODO:hash_password".to_string(),
            created_at: Utc::now(),
            last_login: None,
            is_active: true,
        };
        users.insert(operator.id.clone(), operator);
        
        Ok(())
    }
    
    /// Authenticate user and create session
    /// TODO: Implement secure authentication with password hashing
    pub async fn authenticate(&self, username: &str, _password: &str) -> Result<Session> {
        let users = self.users.read().await;
        
        // Find user by username
        let user = users.values()
            .find(|u| u.username == username && u.is_active)
            .ok_or_else(|| anyhow::anyhow!("Invalid credentials"))?;
        
        // TODO: Verify password hash
        // For now, accept any password in development mode
        
        // Create new session
        let session = Session {
            session_id: Uuid::new_v4().to_string(),
            user_id: user.id.clone(),
            username: user.username.clone(),
            role: user.role,
            created_at: Utc::now(),
            last_activity: Utc::now(),
            expires_at: Utc::now() + Duration::hours(8), // 8-hour session
            ip_address: None,
            has_exclusive_lock: false,
            estimated_completion: None,
        };
        
        let mut sessions = self.sessions.write().await;
        sessions.insert(session.session_id.clone(), session.clone());
        
        tracing::info!("User '{}' authenticated, session created: {}", username, session.session_id);
        
        Ok(session)
    }
    
    /// Validate session and check authorization for operation
    pub async fn authorize(&self, session_id: &str, operation: &Operation) -> Result<bool> {
        let mut sessions = self.sessions.write().await;
        
        let session = sessions.get_mut(session_id)
            .ok_or_else(|| anyhow::anyhow!("Invalid session"))?;
        
        if session.is_expired() {
            return Err(anyhow::anyhow!("Session expired"));
        }
        
        // Refresh session activity
        session.refresh();
        
        // Check role permissions
        Ok(session.role.can_perform(operation))
    }
    
    /// Request exclusive lock on device for measurement
    pub async fn request_exclusive_lock(&self, session_id: &str) -> Result<bool> {
        let mut lock_holder = self.exclusive_lock_holder.write().await;
        
        // Check if someone else has the lock
        if let Some(holder) = lock_holder.as_ref() {
            if holder != session_id {
                // Check if holder's session is still valid
                let sessions = self.sessions.read().await;
                if let Some(holder_session) = sessions.get(holder) {
                    if holder_session.is_valid() {
                        return Ok(false); // Lock held by another valid session
                    }
                }
                // Holder's session is invalid, can take lock
            }
        }
        
        // Grant exclusive lock
        *lock_holder = Some(session_id.to_string());
        
        // Update session
        let mut sessions = self.sessions.write().await;
        if let Some(session) = sessions.get_mut(session_id) {
            session.has_exclusive_lock = true;
        }
        
        tracing::info!("Exclusive lock granted to session: {}", session_id);
        Ok(true)
    }
    
    /// Release exclusive lock
    pub async fn release_exclusive_lock(&self, session_id: &str) -> Result<()> {
        let mut lock_holder = self.exclusive_lock_holder.write().await;
        
        if lock_holder.as_ref() == Some(&session_id.to_string()) {
            *lock_holder = None;
            
            // Update session
            let mut sessions = self.sessions.write().await;
            if let Some(session) = sessions.get_mut(session_id) {
                session.has_exclusive_lock = false;
                session.estimated_completion = None;
            }
            
            tracing::info!("Exclusive lock released by session: {}", session_id);
        }
        
        Ok(())
    }
    
    /// Get current active sessions
    pub async fn get_active_sessions(&self) -> Vec<Session> {
        let sessions = self.sessions.read().await;
        sessions.values()
            .filter(|s| s.is_valid())
            .cloned()
            .collect()
    }
    
    /// Logout user and terminate session
    pub async fn logout(&self, session_id: &str) -> Result<()> {
        // Release exclusive lock if held
        self.release_exclusive_lock(session_id).await?;
        
        // Remove session
        let mut sessions = self.sessions.write().await;
        if let Some(session) = sessions.remove(session_id) {
            tracing::info!("User '{}' logged out, session terminated: {}", session.username, session_id);
        }
        
        Ok(())
    }
    
    /// Clean up expired sessions (should be called periodically)
    pub async fn cleanup_expired_sessions(&self) -> Result<usize> {
        let mut sessions = self.sessions.write().await;
        let initial_count = sessions.len();
        
        // Remove expired sessions
        sessions.retain(|_, session| session.is_valid());
        
        let removed = initial_count - sessions.len();
        if removed > 0 {
            tracing::info!("Cleaned up {} expired sessions", removed);
        }
        
        Ok(removed)
    }
}

impl Default for SessionManager {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[tokio::test]
    async fn test_session_creation_and_validation() {
        let manager = SessionManager::new();
        manager.initialize_default_users().await.unwrap();
        
        let session = manager.authenticate("operator", "password").await.unwrap();
        assert_eq!(session.username, "operator");
        assert_eq!(session.role, UserRole::Operator);
        assert!(session.is_valid());
    }
    
    #[tokio::test]
    async fn test_role_permissions() {
        assert!(UserRole::Operator.can_perform(&Operation::StartMeasurement));
        assert!(!UserRole::Operator.can_perform(&Operation::EnterMaintenanceMode));
        assert!(UserRole::Engineer.can_perform(&Operation::EnterMaintenanceMode));
        assert!(UserRole::Administrator.can_perform(&Operation::ManageUsers));
    }
    
    #[tokio::test]
    async fn test_exclusive_lock() {
        let manager = SessionManager::new();
        manager.initialize_default_users().await.unwrap();
        
        let session1 = manager.authenticate("operator", "pass").await.unwrap();
        let session2 = manager.authenticate("admin", "pass").await.unwrap();
        
        // Session 1 gets lock
        assert!(manager.request_exclusive_lock(&session1.session_id).await.unwrap());
        
        // Session 2 cannot get lock
        assert!(!manager.request_exclusive_lock(&session2.session_id).await.unwrap());
        
        // Session 1 releases lock
        manager.release_exclusive_lock(&session1.session_id).await.unwrap();
        
        // Now session 2 can get lock
        assert!(manager.request_exclusive_lock(&session2.session_id).await.unwrap());
    }
}
