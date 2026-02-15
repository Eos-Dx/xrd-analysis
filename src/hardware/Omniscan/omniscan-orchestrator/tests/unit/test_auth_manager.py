"""
Unit Tests for Authentication Manager
Tests: UR-SAFE-002 (Maintenance Mode Safety), UR-USER-001 (Session Management)
"""

import pytest
from datetime import datetime
import uuid

from omniscan_orchestrator.auth_manager import AuthenticationManager, LoginResult


@pytest.mark.unit
@pytest.mark.safety
class TestKeySwitch Verification:
    """Test key switch verification per UR-SAFE-002."""
    
    @pytest.mark.asyncio
    async def test_login_success_with_key_on(self, auth_manager):
        """Test successful login when key switch is ON."""
        result = await auth_manager.login("OP001", "password123", "clinical_operator")
        
        assert result.success is True
        assert result.session_id is not None
        assert result.error_message is None
    
    @pytest.mark.asyncio
    async def test_login_fails_with_key_off(self, db, mock_grpc_client_key_off):
        """Test login fails when key switch is OFF per UR-SAFE-002."""
        auth_mgr = AuthenticationManager(mock_grpc_client_key_off, db)
        
        result = await auth_mgr.login("OP001", "password123", "clinical_operator")
        
        assert result.success is False
        assert result.session_id is None
        assert "Device locked" in result.error_message
        assert "turn key to operate" in result.error_message
    
    @pytest.mark.asyncio
    async def test_login_handles_grpc_connection_error(self, db, mock_grpc_client):
        """Test graceful handling of gRPC connection errors."""
        import grpc
        
        # Simulate gRPC error
        mock_grpc_client.get_key_switch_state.side_effect = grpc.RpcError()
        auth_mgr = AuthenticationManager(mock_grpc_client, db)
        
        result = await auth_mgr.login("OP001", "password123", "clinical_operator")
        
        assert result.success is False
        assert "Cannot connect to device" in result.error_message


@pytest.mark.unit
class TestSessionManagement:
    """Test session management per UR-USER-001."""
    
    @pytest.mark.asyncio
    async def test_session_creation(self, auth_manager, db):
        """Test session is created and stored in database."""
        result = await auth_manager.login("OP001", "password123", "clinical_operator")
        
        assert result.success is True
        
        # Verify session exists in memory
        session = auth_manager.get_session(result.session_id)
        assert session is not None
        assert session.user_id == "OP001"
        assert session.role == "clinical_operator"
        
        # Verify session stored in database
        db_session = db.get_session(result.session_id)
        assert db_session is not None
        assert db_session.user_id == "OP001"
    
    def test_logout_removes_session(self, auth_manager):
        """Test logout removes session from active sessions."""
        session_id = str(uuid.uuid4())
        
        # Manually add session for testing
        from omniscan_orchestrator.database import UserSession
        session = UserSession(
            session_id=session_id,
            user_id="OP001",
            role="clinical_operator",
            login_time=datetime.utcnow(),
            logout_time=None,
            last_activity=datetime.utcnow()
        )
        auth_manager.active_sessions[session_id] = session
        auth_manager.db_manager.record_login("OP001", session_id, "clinical_operator")
        
        # Logout
        success = auth_manager.logout(session_id)
        
        assert success is True
        assert session_id not in auth_manager.active_sessions
        
        # Verify database updated
        db_session = auth_manager.db_manager.get_session(session_id)
        assert db_session.logout_time is not None
    
    def test_logout_nonexistent_session(self, auth_manager):
        """Test logout of non-existent session returns False."""
        result = auth_manager.logout("invalid-session-id")
        assert result is False
    
    def test_get_session_valid(self, auth_manager):
        """Test retrieving active session."""
        session_id = str(uuid.uuid4())
        
        from omniscan_orchestrator.database import UserSession
        session = UserSession(
            session_id=session_id,
            user_id="OP001",
            role="clinical_operator",
            login_time=datetime.utcnow(),
            logout_time=None,
            last_activity=datetime.utcnow()
        )
        auth_manager.active_sessions[session_id] = session
        
        retrieved = auth_manager.get_session(session_id)
        assert retrieved is not None
        assert retrieved.user_id == "OP001"
    
    def test_get_session_invalid(self, auth_manager):
        """Test retrieving non-existent session returns None."""
        result = auth_manager.get_session("invalid-session")
        assert result is None
    
    def test_is_session_valid(self, auth_manager):
        """Test session validation."""
        session_id = str(uuid.uuid4())
        
        # Should be invalid initially
        assert auth_manager.is_session_valid(session_id) is False
        
        # Add session
        from omniscan_orchestrator.database import UserSession
        session = UserSession(
            session_id=session_id,
            user_id="OP001",
            role="clinical_operator",
            login_time=datetime.utcnow(),
            logout_time=None,
            last_activity=datetime.utcnow()
        )
        auth_manager.active_sessions[session_id] = session
        
        # Should now be valid
        assert auth_manager.is_session_valid(session_id) is True
    
    def test_update_activity(self, auth_manager, db):
        """Test session activity timestamp updates."""
        session_id = str(uuid.uuid4())
        
        from omniscan_orchestrator.database import UserSession
        initial_time = datetime.utcnow()
        session = UserSession(
            session_id=session_id,
            user_id="OP001",
            role="clinical_operator",
            login_time=initial_time,
            logout_time=None,
            last_activity=initial_time
        )
        auth_manager.active_sessions[session_id] = session
        db.record_login("OP001", session_id, "clinical_operator")
        
        # Wait and update
        import time
        time.sleep(0.1)
        auth_manager.update_activity(session_id)
        
        # Check in-memory session
        updated_session = auth_manager.get_session(session_id)
        assert updated_session.last_activity > initial_time
        
        # Check database
        db_session = db.get_session(session_id)
        assert db_session.last_activity > initial_time
    
    def test_require_session_valid(self, auth_manager):
        """Test require_session with valid session."""
        session_id = str(uuid.uuid4())
        
        from omniscan_orchestrator.database import UserSession
        session = UserSession(
            session_id=session_id,
            user_id="OP001",
            role="clinical_operator",
            login_time=datetime.utcnow(),
            logout_time=None,
            last_activity=datetime.utcnow()
        )
        auth_manager.active_sessions[session_id] = session
        
        success, error = auth_manager.require_session(session_id)
        
        assert success is True
        assert error is None
    
    def test_require_session_invalid(self, auth_manager):
        """Test require_session with invalid session."""
        success, error = auth_manager.require_session("invalid-session")
        
        assert success is False
        assert "Invalid or expired session" in error
    
    def test_get_user_id_from_session(self, auth_manager):
        """Test retrieving user ID from session."""
        session_id = str(uuid.uuid4())
        
        from omniscan_orchestrator.database import UserSession
        session = UserSession(
            session_id=session_id,
            user_id="OP001",
            role="clinical_operator",
            login_time=datetime.utcnow(),
            logout_time=None,
            last_activity=datetime.utcnow()
        )
        auth_manager.active_sessions[session_id] = session
        
        user_id = auth_manager.get_user_id(session_id)
        assert user_id == "OP001"
    
    def test_get_user_role_from_session(self, auth_manager):
        """Test retrieving user role from session."""
        session_id = str(uuid.uuid4())
        
        from omniscan_orchestrator.database import UserSession
        session = UserSession(
            session_id=session_id,
            user_id="OP001",
            role="clinical_operator",
            login_time=datetime.utcnow(),
            logout_time=None,
            last_activity=datetime.utcnow()
        )
        auth_manager.active_sessions[session_id] = session
        
        role = auth_manager.get_user_role(session_id)
        assert role == "clinical_operator"


@pytest.mark.unit
class TestAuthenticationSecurity:
    """Test authentication security features."""
    
    @pytest.mark.asyncio
    async def test_unique_session_ids(self, auth_manager):
        """Test that each login generates unique session ID."""
        result1 = await auth_manager.login("OP001", "pass1", "clinical_operator")
        result2 = await auth_manager.login("OP002", "pass2", "clinical_operator")
        
        assert result1.session_id != result2.session_id
    
    @pytest.mark.asyncio
    async def test_concurrent_user_sessions(self, auth_manager):
        """Test multiple users can have concurrent sessions."""
        result1 = await auth_manager.login("OP001", "pass1", "clinical_operator")
        result2 = await auth_manager.login("ENG001", "pass2", "maintenance_engineer")
        
        assert result1.success is True
        assert result2.success is True
        
        # Both sessions should be active
        assert auth_manager.is_session_valid(result1.session_id)
        assert auth_manager.is_session_valid(result2.session_id)
    
    def test_session_isolation(self, auth_manager):
        """Test that sessions are isolated per user."""
        session1_id = str(uuid.uuid4())
        session2_id = str(uuid.uuid4())
        
        from omniscan_orchestrator.database import UserSession
        
        session1 = UserSession(
            session_id=session1_id,
            user_id="OP001",
            role="clinical_operator",
            login_time=datetime.utcnow(),
            logout_time=None,
            last_activity=datetime.utcnow()
        )
        
        session2 = UserSession(
            session_id=session2_id,
            user_id="ENG001",
            role="maintenance_engineer",
            login_time=datetime.utcnow(),
            logout_time=None,
            last_activity=datetime.utcnow()
        )
        
        auth_manager.active_sessions[session1_id] = session1
        auth_manager.active_sessions[session2_id] = session2
        
        # Verify isolation
        assert auth_manager.get_user_id(session1_id) == "OP001"
        assert auth_manager.get_user_id(session2_id) == "ENG001"
        assert auth_manager.get_user_role(session1_id) != auth_manager.get_user_role(session2_id)


@pytest.mark.unit
class TestCredentialVerification:
    """Test credential verification (stub implementation)."""
    
    def test_verify_credentials_stub(self, auth_manager):
        """Test that credential verification stub accepts all credentials."""
        # Note: This tests the current stub implementation
        # In production, this should be replaced with real authentication
        assert auth_manager._verify_credentials("any_user", "any_password") is True
    
    def test_verify_credentials_documentation(self, auth_manager):
        """Verify that _verify_credentials has TODO for production implementation."""
        import inspect
        source = inspect.getsource(auth_manager._verify_credentials)
        
        assert "TODO" in source or "STUB" in source
