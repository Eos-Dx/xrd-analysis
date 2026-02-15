"""
Unit Tests for Role-Based Access Control (RBAC)
Tests: UR-USER-002 (Role-Based Access Control)
"""

import pytest
from datetime import datetime, timezone

from omniscan_orchestrator.rbac import (
    Role,
    Permission,
    User,
    AccessControlError,
    require_permission,
    get_user_permissions,
    ROLE_PERMISSIONS
)


@pytest.mark.unit
@pytest.mark.rbac
class TestUserCreation:
    """Test user creation and role assignment."""
    
    def test_create_clinical_operator(self):
        """Test creating clinical operator user."""
        user = User(
            user_id="OP001",
            username="operator:OP001",
            role=Role.CLINICAL_OPERATOR,
            certificate_cn="operator:OP001",
            authenticated_at=datetime.now(timezone.utc)
        )
        
        assert user.user_id == "OP001"
        assert user.role == Role.CLINICAL_OPERATOR
        assert user.get_role_name() == "Clinical Operator"
    
    def test_create_maintenance_engineer(self):
        """Test creating maintenance engineer user."""
        user = User(
            user_id="ENG001",
            username="engineer:ENG001",
            role=Role.MAINTENANCE_ENGINEER
        )
        
        assert user.role == Role.MAINTENANCE_ENGINEER
        assert user.get_role_name() == "Maintenance Engineer"
    
    def test_create_administrator(self):
        """Test creating administrator user."""
        user = User(
            user_id="ADMIN001",
            username="admin:ADMIN001",
            role=Role.ADMINISTRATOR
        )
        
        assert user.role == Role.ADMINISTRATOR
        assert user.get_role_name() == "Administrator"
    
    def test_from_certificate(self):
        """Test user creation from mTLS certificate."""
        # Engineer
        user = User.from_certificate("engineer:ENG001")
        assert user is not None
        assert user.role == Role.MAINTENANCE_ENGINEER
        assert user.user_id == "ENG001"
        
        # Operator
        user = User.from_certificate("operator:OP001")
        assert user is not None
        assert user.role == Role.CLINICAL_OPERATOR
        
        # Admin
        user = User.from_certificate("admin:ADMIN001")
        assert user is not None
        assert user.role == Role.ADMINISTRATOR
    
    def test_from_certificate_invalid_format(self):
        """Test that invalid certificate format returns None."""
        user = User.from_certificate("invalid_format")
        assert user is None
        
        user = User.from_certificate("unknown:USER001")
        assert user is None


@pytest.mark.unit
@pytest.mark.rbac
class TestClinicalOperatorPermissions:
    """Test clinical operator permissions per UR-USER-002."""
    
    def test_can_execute_measurement(self, clinical_operator):
        """Test clinical operator can execute measurements."""
        assert clinical_operator.has_permission(Permission.EXECUTE_MEASUREMENT)
    
    def test_can_perform_calibration(self, clinical_operator):
        """Test clinical operator can perform calibration."""
        assert clinical_operator.has_permission(Permission.PERFORM_CALIBRATION)
    
    def test_can_view_results(self, clinical_operator):
        """Test clinical operator can view measurement results."""
        assert clinical_operator.has_permission(Permission.VIEW_MEASUREMENT_RESULTS)
        assert clinical_operator.has_permission(Permission.VIEW_CALIBRATION_DATA)
    
    def test_can_view_config(self, clinical_operator):
        """Test clinical operator can view configuration."""
        assert clinical_operator.has_permission(Permission.VIEW_CONFIG)
    
    def test_cannot_modify_config(self, clinical_operator):
        """Test clinical operator cannot modify configuration."""
        assert not clinical_operator.has_permission(Permission.MODIFY_CONFIG)
    
    def test_cannot_enter_maintenance(self, clinical_operator):
        """Test clinical operator cannot enter maintenance mode."""
        assert not clinical_operator.has_permission(Permission.ENTER_MAINTENANCE_MODE)
    
    def test_cannot_manage_users(self, clinical_operator):
        """Test clinical operator cannot manage users."""
        assert not clinical_operator.has_permission(Permission.MANAGE_USERS)
    
    def test_cannot_view_audit_logs(self, clinical_operator):
        """Test clinical operator cannot view audit logs."""
        assert not clinical_operator.has_permission(Permission.VIEW_AUDIT_LOGS)


@pytest.mark.unit
@pytest.mark.rbac
class TestMaintenanceEngineerPermissions:
    """Test maintenance engineer permissions per UR-USER-002."""
    
    def test_has_all_operator_permissions(self, maintenance_engineer):
        """Test engineer has all operator permissions."""
        assert maintenance_engineer.has_permission(Permission.EXECUTE_MEASUREMENT)
        assert maintenance_engineer.has_permission(Permission.PERFORM_CALIBRATION)
        assert maintenance_engineer.has_permission(Permission.VIEW_MEASUREMENT_RESULTS)
    
    def test_can_modify_config(self, maintenance_engineer):
        """Test engineer can modify configuration."""
        assert maintenance_engineer.has_permission(Permission.MODIFY_CONFIG)
    
    def test_can_enter_maintenance(self, maintenance_engineer):
        """Test engineer can enter maintenance mode."""
        assert maintenance_engineer.has_permission(Permission.ENTER_MAINTENANCE_MODE)
        assert maintenance_engineer.has_permission(Permission.EXIT_MAINTENANCE_MODE)
    
    def test_can_bypass_safety(self, maintenance_engineer):
        """Test engineer can bypass safety checks per UR-SAFE-002."""
        assert maintenance_engineer.has_permission(Permission.BYPASS_SAFETY_CHECKS)
    
    def test_can_view_audit_logs(self, maintenance_engineer):
        """Test engineer can view audit logs."""
        assert maintenance_engineer.has_permission(Permission.VIEW_AUDIT_LOGS)
    
    def test_can_view_sessions(self, maintenance_engineer):
        """Test engineer can view active sessions."""
        assert maintenance_engineer.has_permission(Permission.VIEW_SESSIONS)
    
    def test_cannot_manage_users(self, maintenance_engineer):
        """Test engineer cannot manage users."""
        assert not maintenance_engineer.has_permission(Permission.MANAGE_USERS)
    
    def test_cannot_export_audit_logs(self, maintenance_engineer):
        """Test engineer cannot export audit logs."""
        assert not maintenance_engineer.has_permission(Permission.EXPORT_AUDIT_LOGS)


@pytest.mark.unit
@pytest.mark.rbac
class TestAdministratorPermissions:
    """Test administrator permissions per UR-USER-002."""
    
    def test_can_manage_users(self, administrator):
        """Test admin can manage users."""
        assert administrator.has_permission(Permission.MANAGE_USERS)
    
    def test_can_view_and_export_audit_logs(self, administrator):
        """Test admin can view and export audit logs."""
        assert administrator.has_permission(Permission.VIEW_AUDIT_LOGS)
        assert administrator.has_permission(Permission.EXPORT_AUDIT_LOGS)
    
    def test_can_view_sessions(self, administrator):
        """Test admin can view sessions."""
        assert administrator.has_permission(Permission.VIEW_SESSIONS)
        assert administrator.has_permission(Permission.TERMINATE_SESSIONS)
    
    def test_can_view_config(self, administrator):
        """Test admin can view configuration."""
        assert administrator.has_permission(Permission.VIEW_CONFIG)
        assert administrator.has_permission(Permission.MODIFY_CONFIG)
    
    def test_can_view_results(self, administrator):
        """Test admin can view results."""
        assert administrator.has_permission(Permission.VIEW_MEASUREMENT_RESULTS)
        assert administrator.has_permission(Permission.VIEW_CALIBRATION_DATA)
    
    def test_cannot_execute_measurement(self, administrator):
        """Test admin cannot execute measurements."""
        assert not administrator.has_permission(Permission.EXECUTE_MEASUREMENT)
    
    def test_cannot_enter_maintenance(self, administrator):
        """Test admin cannot enter maintenance mode."""
        assert not administrator.has_permission(Permission.ENTER_MAINTENANCE_MODE)


@pytest.mark.unit
@pytest.mark.rbac
class TestAccessControlEnforcement:
    """Test access control enforcement per UR-USER-002."""
    
    def test_require_permission_success(self, maintenance_engineer):
        """Test permission check passes for authorized user."""
        # Should not raise exception
        require_permission(maintenance_engineer, Permission.MODIFY_CONFIG)
    
    def test_require_permission_failure(self, clinical_operator):
        """Test permission check fails for unauthorized user."""
        with pytest.raises(AccessControlError) as excinfo:
            require_permission(clinical_operator, Permission.MODIFY_CONFIG)
        
        assert "Access denied" in str(excinfo.value)
        assert "does not have permission" in str(excinfo.value)
    
    def test_access_control_error_details(self, clinical_operator):
        """Test AccessControlError contains detailed information."""
        try:
            require_permission(
                clinical_operator,
                Permission.MANAGE_USERS,
                reason="User management requires admin role"
            )
            assert False, "Should have raised AccessControlError"
        except AccessControlError as e:
            assert e.user == clinical_operator
            assert e.permission == Permission.MANAGE_USERS
            assert "User management requires admin role" in e.reason
    
    def test_get_user_permissions(self, maintenance_engineer):
        """Test retrieving all permissions for a user."""
        permissions = get_user_permissions(maintenance_engineer)
        
        assert isinstance(permissions, set)
        assert Permission.MODIFY_CONFIG in permissions
        assert Permission.ENTER_MAINTENANCE_MODE in permissions
        assert Permission.EXECUTE_MEASUREMENT in permissions
        
        # Should not have admin-only permissions
        assert Permission.MANAGE_USERS not in permissions


@pytest.mark.unit
@pytest.mark.rbac
class TestRolePermissionMappings:
    """Test that role-permission mappings are correctly defined."""
    
    def test_all_roles_have_mappings(self):
        """Test all roles have permission mappings."""
        for role in Role:
            assert role in ROLE_PERMISSIONS
            assert isinstance(ROLE_PERMISSIONS[role], set)
    
    def test_operator_subset_of_engineer(self):
        """Test that operator permissions are subset of engineer."""
        operator_perms = ROLE_PERMISSIONS[Role.CLINICAL_OPERATOR]
        engineer_perms = ROLE_PERMISSIONS[Role.MAINTENANCE_ENGINEER]
        
        assert operator_perms.issubset(engineer_perms)
    
    def test_no_overlap_admin_engineer_exclusive(self):
        """Test admin and engineer have distinct exclusive permissions."""
        admin_perms = ROLE_PERMISSIONS[Role.ADMINISTRATOR]
        engineer_perms = ROLE_PERMISSIONS[Role.MAINTENANCE_ENGINEER]
        
        # Admin should have MANAGE_USERS (exclusive)
        assert Permission.MANAGE_USERS in admin_perms
        assert Permission.MANAGE_USERS not in engineer_perms
        
        # Engineer should have ENTER_MAINTENANCE_MODE (exclusive)
        assert Permission.ENTER_MAINTENANCE_MODE in engineer_perms
        assert Permission.ENTER_MAINTENANCE_MODE not in admin_perms
    
    def test_safety_critical_permissions_restricted(self):
        """Test safety-critical permissions are properly restricted."""
        # Only engineers should bypass safety
        assert Permission.BYPASS_SAFETY_CHECKS in ROLE_PERMISSIONS[Role.MAINTENANCE_ENGINEER]
        assert Permission.BYPASS_SAFETY_CHECKS not in ROLE_PERMISSIONS[Role.CLINICAL_OPERATOR]
        assert Permission.BYPASS_SAFETY_CHECKS not in ROLE_PERMISSIONS[Role.ADMINISTRATOR]


@pytest.mark.unit
@pytest.mark.rbac
class TestPermissionSeparation:
    """Test separation of duties per FDA requirements."""
    
    def test_measurement_execution_separation(self, administrator, clinical_operator):
        """Test admin cannot execute measurements (separation of duties)."""
        assert not administrator.has_permission(Permission.EXECUTE_MEASUREMENT)
        assert clinical_operator.has_permission(Permission.EXECUTE_MEASUREMENT)
    
    def test_audit_log_access_separation(self, clinical_operator, administrator):
        """Test only admin can export audit logs."""
        assert not clinical_operator.has_permission(Permission.EXPORT_AUDIT_LOGS)
        assert administrator.has_permission(Permission.EXPORT_AUDIT_LOGS)
    
    def test_user_management_separation(self, maintenance_engineer, administrator):
        """Test only admin can manage users."""
        assert not maintenance_engineer.has_permission(Permission.MANAGE_USERS)
        assert administrator.has_permission(Permission.MANAGE_USERS)
    
    def test_maintenance_mode_separation(self, administrator, maintenance_engineer):
        """Test only engineer can enter maintenance mode."""
        assert not administrator.has_permission(Permission.ENTER_MAINTENANCE_MODE)
        assert maintenance_engineer.has_permission(Permission.ENTER_MAINTENANCE_MODE)


@pytest.mark.unit
@pytest.mark.rbac
class TestAuthenticationTimestamp:
    """Test authentication timestamp tracking."""
    
    def test_authenticated_at_recorded(self):
        """Test authentication timestamp is recorded."""
        auth_time = datetime.now(timezone.utc)
        user = User(
            user_id="OP001",
            username="operator:OP001",
            role=Role.CLINICAL_OPERATOR,
            authenticated_at=auth_time
        )
        
        assert user.authenticated_at == auth_time
    
    def test_from_certificate_sets_timestamp(self):
        """Test from_certificate sets authentication timestamp."""
        before = datetime.now(timezone.utc)
        user = User.from_certificate("operator:OP001")
        after = datetime.now(timezone.utc)
        
        assert user.authenticated_at is not None
        assert before <= user.authenticated_at <= after
