"""
Role-Based Access Control (RBAC) Module

Implements access control as required by USER_EXPECTATIONS.md Section 7.

User Roles:
- Clinical Operator: Measurement execution and calibration
- Maintenance Engineer: Full diagnostic access and maintenance mode
- Administrator: User management and system configuration
"""

from __future__ import annotations

from enum import Enum
from typing import Optional, Set
from dataclasses import dataclass
from datetime import datetime, timezone


class Role(Enum):
    """User roles as defined in USER_EXPECTATIONS.md Section 7."""
    CLINICAL_OPERATOR = "clinical_operator"
    MAINTENANCE_ENGINEER = "maintenance_engineer"
    ADMINISTRATOR = "administrator"
    
    
class Permission(Enum):
    """System permissions."""
    # Measurement operations
    EXECUTE_MEASUREMENT = "execute_measurement"
    VIEW_MEASUREMENT_RESULTS = "view_measurement_results"
    
    # Calibration
    PERFORM_CALIBRATION = "perform_calibration"
    VIEW_CALIBRATION_DATA = "view_calibration_data"
    
    # Device control
    CONTROL_DEVICE_POWER = "control_device_power"
    VIEW_DEVICE_STATE = "view_device_state"
    
    # Configuration
    VIEW_CONFIG = "view_config"
    MODIFY_CONFIG = "modify_config"
    
    # Maintenance
    ENTER_MAINTENANCE_MODE = "enter_maintenance_mode"
    EXIT_MAINTENANCE_MODE = "exit_maintenance_mode"
    BYPASS_SAFETY_CHECKS = "bypass_safety_checks"
    
    # User management
    MANAGE_USERS = "manage_users"
    VIEW_AUDIT_LOGS = "view_audit_logs"
    EXPORT_AUDIT_LOGS = "export_audit_logs"
    
    # Session monitoring
    VIEW_SESSIONS = "view_sessions"
    TERMINATE_SESSIONS = "terminate_sessions"


# Role-to-Permission mapping as per USER_EXPECTATIONS.md
ROLE_PERMISSIONS: dict[Role, Set[Permission]] = {
    Role.CLINICAL_OPERATOR: {
        Permission.EXECUTE_MEASUREMENT,
        Permission.VIEW_MEASUREMENT_RESULTS,
        Permission.PERFORM_CALIBRATION,
        Permission.VIEW_CALIBRATION_DATA,
        Permission.VIEW_DEVICE_STATE,
        Permission.CONTROL_DEVICE_POWER,
        Permission.VIEW_CONFIG,
    },
    Role.MAINTENANCE_ENGINEER: {
        Permission.EXECUTE_MEASUREMENT,
        Permission.VIEW_MEASUREMENT_RESULTS,
        Permission.PERFORM_CALIBRATION,
        Permission.VIEW_CALIBRATION_DATA,
        Permission.VIEW_DEVICE_STATE,
        Permission.CONTROL_DEVICE_POWER,
        Permission.VIEW_CONFIG,
        Permission.MODIFY_CONFIG,
        Permission.ENTER_MAINTENANCE_MODE,
        Permission.EXIT_MAINTENANCE_MODE,
        Permission.BYPASS_SAFETY_CHECKS,
        Permission.VIEW_AUDIT_LOGS,
        Permission.VIEW_SESSIONS,
    },
    Role.ADMINISTRATOR: {
        Permission.VIEW_MEASUREMENT_RESULTS,
        Permission.VIEW_CALIBRATION_DATA,
        Permission.VIEW_DEVICE_STATE,
        Permission.VIEW_CONFIG,
        Permission.MODIFY_CONFIG,
        Permission.MANAGE_USERS,
        Permission.VIEW_AUDIT_LOGS,
        Permission.EXPORT_AUDIT_LOGS,
        Permission.VIEW_SESSIONS,
        Permission.TERMINATE_SESSIONS,
    },
}


@dataclass
class User:
    """User with role and authentication info."""
    user_id: str
    username: str
    role: Role
    certificate_cn: Optional[str] = None  # From mTLS certificate
    authenticated_at: Optional[datetime] = None
    
    def has_permission(self, permission: Permission) -> bool:
        """Check if user has a specific permission."""
        return permission in ROLE_PERMISSIONS.get(self.role, set())
        
    def get_role_name(self) -> str:
        """Get human-readable role name."""
        return {
            Role.CLINICAL_OPERATOR: "Clinical Operator",
            Role.MAINTENANCE_ENGINEER: "Maintenance Engineer",
            Role.ADMINISTRATOR: "Administrator",
        }[self.role]
        
    @classmethod
    def from_certificate(cls, cert_common_name: str) -> Optional[User]:
        """
        Extract user from mTLS certificate.
        
        Certificate CN format expected:
        - "engineer:<engineer_id>" -> Maintenance Engineer
        - "operator:<operator_id>" -> Clinical Operator  
        - "admin:<admin_id>" -> Administrator
        """
        if not cert_common_name:
            return None
            
        parts = cert_common_name.split(":", 1)
        if len(parts) != 2:
            return None
            
        role_type, user_id = parts
        
        role_map = {
            "engineer": Role.MAINTENANCE_ENGINEER,
            "operator": Role.CLINICAL_OPERATOR,
            "admin": Role.ADMINISTRATOR,
        }
        
        role = role_map.get(role_type.lower())
        if not role:
            return None
            
        return cls(
            user_id=user_id,
            username=cert_common_name,
            role=role,
            certificate_cn=cert_common_name,
            authenticated_at=datetime.now(timezone.utc),
        )


class AccessControlError(Exception):
    """Raised when access is denied."""
    
    def __init__(self, user: User, permission: Permission, reason: str = ""):
        self.user = user
        self.permission = permission
        self.reason = reason
        super().__init__(
            f"Access denied: {user.username} ({user.get_role_name()}) "
            f"does not have permission {permission.value}"
            + (f": {reason}" if reason else "")
        )


def require_permission(user: User, permission: Permission, reason: str = ""):
    """
    Enforce permission requirement. Raises AccessControlError if denied.
    
    Args:
        user: The user attempting the operation
        permission: Required permission
        reason: Additional context for denial
        
    Raises:
        AccessControlError: If user lacks permission
    """
    if not user.has_permission(permission):
        raise AccessControlError(user, permission, reason)


def get_user_permissions(user: User) -> Set[Permission]:
    """Get all permissions for a user."""
    return ROLE_PERMISSIONS.get(user.role, set())


def format_permissions(permissions: Set[Permission]) -> str:
    """Format permissions for display."""
    return "\n".join(f"  • {p.value}" for p in sorted(permissions, key=lambda x: x.value))
