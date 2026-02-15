use omniscan_hw_server::auth::*;

#[tokio::test]
async fn test_session_manager_initialization() {
    let manager = SessionManager::new();
    manager.initialize_default_users().await.unwrap();
    
    let sessions = manager.get_active_sessions().await;
    assert_eq!(sessions.len(), 0);
}

#[tokio::test]
async fn test_authentication() {
    let manager = SessionManager::new();
    manager.initialize_default_users().await.unwrap();
    
    let session = manager.authenticate("operator", "password").await.unwrap();
    assert_eq!(session.username, "operator");
    assert_eq!(session.role, UserRole::Operator);
    assert!(session.is_valid());
    assert!(!session.has_exclusive_lock);
}

#[tokio::test]
async fn test_invalid_username() {
    let manager = SessionManager::new();
    manager.initialize_default_users().await.unwrap();
    
    let result = manager.authenticate("nonexistent", "password").await;
    assert!(result.is_err());
}

#[tokio::test]
async fn test_role_permissions() {
    // Operators can start measurements
    assert!(UserRole::Operator.can_perform(&Operation::StartMeasurement));
    assert!(UserRole::Operator.can_perform(&Operation::RunCalibration));
    
    // Operators cannot enter maintenance mode
    assert!(!UserRole::Operator.can_perform(&Operation::EnterMaintenanceMode));
    assert!(!UserRole::Operator.can_perform(&Operation::ManageUsers));
    
    // Engineers can do maintenance
    assert!(UserRole::Engineer.can_perform(&Operation::EnterMaintenanceMode));
    assert!(UserRole::Engineer.can_perform(&Operation::AccessDiagnostics));
    
    // Only administrators can manage users
    assert!(!UserRole::Operator.can_perform(&Operation::ManageUsers));
    assert!(!UserRole::Engineer.can_perform(&Operation::ManageUsers));
    assert!(UserRole::Administrator.can_perform(&Operation::ManageUsers));
}

#[tokio::test]
async fn test_authorization() {
    let manager = SessionManager::new();
    manager.initialize_default_users().await.unwrap();
    
    let session = manager.authenticate("operator", "password").await.unwrap();
    
    // Should be authorized for measurements
    let authorized = manager.authorize(&session.session_id, &Operation::StartMeasurement).await.unwrap();
    assert!(authorized);
    
    // Should not be authorized for user management
    let authorized = manager.authorize(&session.session_id, &Operation::ManageUsers).await.unwrap();
    assert!(!authorized);
}

#[tokio::test]
async fn test_exclusive_lock() {
    let manager = SessionManager::new();
    manager.initialize_default_users().await.unwrap();
    
    let session1 = manager.authenticate("operator", "password").await.unwrap();
    let session2 = manager.authenticate("admin", "password").await.unwrap();
    
    // Session 1 gets lock
    let got_lock = manager.request_exclusive_lock(&session1.session_id).await.unwrap();
    assert!(got_lock);
    
    // Session 2 cannot get lock
    let got_lock = manager.request_exclusive_lock(&session2.session_id).await.unwrap();
    assert!(!got_lock);
    
    // Session 1 releases lock
    manager.release_exclusive_lock(&session1.session_id).await.unwrap();
    
    // Now session 2 can get lock
    let got_lock = manager.request_exclusive_lock(&session2.session_id).await.unwrap();
    assert!(got_lock);
}

#[tokio::test]
async fn test_session_expiry() {
    let manager = SessionManager::new();
    manager.initialize_default_users().await.unwrap();
    
    let session = manager.authenticate("operator", "password").await.unwrap();
    assert!(session.is_valid());
    assert!(!session.is_expired());
}

#[tokio::test]
async fn test_logout() {
    let manager = SessionManager::new();
    manager.initialize_default_users().await.unwrap();
    
    let session = manager.authenticate("operator", "password").await.unwrap();
    let session_id = session.session_id.clone();
    
    let sessions = manager.get_active_sessions().await;
    assert_eq!(sessions.len(), 1);
    
    manager.logout(&session_id).await.unwrap();
    
    let sessions = manager.get_active_sessions().await;
    assert_eq!(sessions.len(), 0);
}

#[tokio::test]
async fn test_multiple_active_sessions() {
    let manager = SessionManager::new();
    manager.initialize_default_users().await.unwrap();
    
    let _session1 = manager.authenticate("operator", "password").await.unwrap();
    let _session2 = manager.authenticate("admin", "password").await.unwrap();
    
    let sessions = manager.get_active_sessions().await;
    assert_eq!(sessions.len(), 2);
}

#[tokio::test]
async fn test_exclusive_lock_released_on_logout() {
    let manager = SessionManager::new();
    manager.initialize_default_users().await.unwrap();
    
    let session1 = manager.authenticate("operator", "password").await.unwrap();
    let session2 = manager.authenticate("admin", "password").await.unwrap();
    
    // Session 1 gets lock
    manager.request_exclusive_lock(&session1.session_id).await.unwrap();
    
    // Logout releases lock
    manager.logout(&session1.session_id).await.unwrap();
    
    // Session 2 can now get lock
    let got_lock = manager.request_exclusive_lock(&session2.session_id).await.unwrap();
    assert!(got_lock);
}

#[test]
fn test_all_roles_cover_all_operations() {
    let admin = UserRole::Administrator;
    
    // Admin should be able to do everything
    assert!(admin.can_perform(&Operation::StartMeasurement));
    assert!(admin.can_perform(&Operation::StopMeasurement));
    assert!(admin.can_perform(&Operation::RunCalibration));
    assert!(admin.can_perform(&Operation::ViewMeasurementData));
    assert!(admin.can_perform(&Operation::EnterMaintenanceMode));
    assert!(admin.can_perform(&Operation::AccessDiagnostics));
    assert!(admin.can_perform(&Operation::BypassInterlocks));
    assert!(admin.can_perform(&Operation::ManageUsers));
    assert!(admin.can_perform(&Operation::ModifyConfiguration));
    assert!(admin.can_perform(&Operation::ViewAuditLogs));
}
