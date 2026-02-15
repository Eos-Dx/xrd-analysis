"""
Authentication Manager for Omniscan Orchestrator

Handles user authentication with hardware server key switch verification,
session management, and login/logout operations.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Optional, Tuple, Dict
import uuid
import grpc

from .database import OrchestratorDatabase, UserSession


@dataclass
class LoginResult:
    """Result of login attempt."""
    success: bool
    session_id: Optional[str]
    error_message: Optional[str]


class AuthenticationManager:
    """Manages user authentication and session lifecycle.
    
    Verifies key switch state before allowing login to ensure
    physical security control is engaged.
    """
    
    def __init__(self, grpc_client, db_manager: OrchestratorDatabase):
        """Initialize authentication manager.
        
        Args:
            grpc_client: gRPC client for hardware server communication
            db_manager: Database manager for session storage
        """
        self.grpc_client = grpc_client
        self.db_manager = db_manager
        self.active_sessions: Dict[str, UserSession] = {}
    
    async def login(
        self,
        user_id: str,
        password: str,
        role: str
    ) -> LoginResult:
        """Authenticate user and create session.
        
        Process:
        1. Verify user credentials (local or LDAP)
        2. Check hardware key switch state via gRPC
        3. Create session if both checks pass
        4. Store session in database
        
        Args:
            user_id: User identifier
            password: User password
            role: User role ("operator", "engineer", "admin")
            
        Returns:
            LoginResult with success status, session_id, or error message
        """
        # Step 1: Verify credentials
        if not self._verify_credentials(user_id, password):
            return LoginResult(
                success=False,
                session_id=None,
                error_message="Invalid credentials"
            )
        
        # Step 2: Check key switch state via gRPC (use GPIO state for portability)
        try:
            gpio_state = self.grpc_client.get_gpio_state()
            if isinstance(gpio_state, dict) and gpio_state.get("error"):
                return LoginResult(
                    success=False,
                    session_id=None,
                    error_message=f"Cannot connect to device: {gpio_state['error']}"
                )

            key_on = bool(gpio_state.get("key_switch_on", False))
            if not key_on:
                return LoginResult(
                    success=False,
                    session_id=None,
                    error_message="Device locked - turn key to operate"
                )
                
        except grpc.RpcError as e:
            return LoginResult(
                success=False,
                session_id=None,
                error_message=f"Cannot connect to device: {e.details()}"
            )
        except Exception as e:
            return LoginResult(
                success=False,
                session_id=None,
                error_message=f"Connection error: {str(e)}"
            )
        
        # Step 3: Create session
        session_id = str(uuid.uuid4())
        now = datetime.utcnow()
        
        session = UserSession(
            session_id=session_id,
            user_id=user_id,
            role=role,
            login_time=now,
            logout_time=None,
            last_activity=now
        )
        
        # Step 4: Store session
        self.active_sessions[session_id] = session
        self.db_manager.record_login(user_id, session_id, role)
        
        return LoginResult(
            success=True,
            session_id=session_id,
            error_message=None
        )
    
    def _verify_credentials(self, user_id: str, password: str) -> bool:
        """Verify user credentials.
        
        TODO: Implement real authentication (LDAP, database, etc.)
        For now, this is a stub that accepts all credentials for development.
        
        Args:
            user_id: User identifier
            password: User password
            
        Returns:
            True if credentials are valid, False otherwise
        """
        # STUB: Accept all credentials for development
        # In production, implement:
        # - Database lookup for local users
        # - LDAP/Active Directory integration
        # - Password hashing verification
        # - Account lockout after failed attempts
        return True
    
    def logout(self, session_id: str) -> bool:
        """Logout user and clean up session.
        
        Args:
            session_id: Session identifier to logout
            
        Returns:
            True if logout successful, False if session not found
        """
        if session_id not in self.active_sessions:
            return False
        
        # Remove from active sessions
        del self.active_sessions[session_id]
        
        # Update database
        self.db_manager.record_logout(session_id)
        
        return True
    
    def get_session(self, session_id: str) -> Optional[UserSession]:
        """Get active session by ID.
        
        Args:
            session_id: Session identifier
            
        Returns:
            UserSession if found and active, None otherwise
        """
        return self.active_sessions.get(session_id)
    
    def update_activity(self, session_id: str):
        """Update last activity time for session.
        
        Args:
            session_id: Session identifier
        """
        if session_id in self.active_sessions:
            self.active_sessions[session_id].last_activity = datetime.utcnow()
            self.db_manager.update_session_activity(session_id)
    
    def is_session_valid(self, session_id: str) -> bool:
        """Check if session is valid and active.
        
        Args:
            session_id: Session identifier
            
        Returns:
            True if session exists and is active, False otherwise
        """
        return session_id in self.active_sessions
    
    def require_session(self, session_id: str) -> Tuple[bool, Optional[str]]:
        """Validate session is active.
        
        Args:
            session_id: Session identifier
            
        Returns:
            Tuple of (success, error_message)
        """
        if not self.is_session_valid(session_id):
            return False, "Invalid or expired session - please login"
        
        # Update activity
        self.update_activity(session_id)
        
        return True, None
    
    def get_user_id(self, session_id: str) -> Optional[str]:
        """Get user ID from session.
        
        Args:
            session_id: Session identifier
            
        Returns:
            User ID if session is valid, None otherwise
        """
        session = self.get_session(session_id)
        return session.user_id if session else None
    
    def get_user_role(self, session_id: str) -> Optional[str]:
        """Get user role from session.
        
        Args:
            session_id: Session identifier
            
        Returns:
            User role if session is valid, None otherwise
        """
        session = self.get_session(session_id)
        return session.role if session else None
