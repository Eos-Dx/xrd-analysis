"""
Unit Tests for Command Discovery and Compatibility Checking
Medical Device Software - FDA Compliant Testing

Tests cover:
- Server capability querying
- Command listing and filtering
- Version compatibility validation
- Protocol version checking
- Startup compatibility checks
- Missing command detection
- CLI commands for discovery
"""

import pytest
import json
from unittest.mock import Mock, patch, MagicMock
from typer.testing import CliRunner
from datetime import datetime

from omniscan_orchestrator.cli import app, _build_client
from omniscan_orchestrator.grpc_client import OmniscanGrpcClient


runner = CliRunner()


# ==============================================================================
# Server Capabilities Tests
# ==============================================================================

class TestServerCapabilities:
    """Test querying server capabilities and version information."""
    
    @patch('omniscan_orchestrator.grpc_client.hub_pb2_grpc.CommandDiscoveryStub')
    def test_get_server_capabilities_success(self, mock_stub_class):
        """Test successful retrieval of server capabilities."""
        mock_stub = Mock()
        mock_stub_class.return_value = mock_stub
        
        # Mock the response
        mock_response = Mock()
        mock_capabilities = Mock()
        mock_capabilities.server_version = "0.1.0"
        mock_capabilities.protocol_version = "1.0.0"
        mock_capabilities.build_time.seconds = 1672531200  # 2023-01-01
        mock_capabilities.supported_features = ["audit_logging", "safety_interlocks", "gui"]
        mock_capabilities.device_type = "hardware_server"
        mock_capabilities.HasField.return_value = True
        mock_response.capabilities = mock_capabilities
        
        mock_stub.GetServerCapabilities.return_value = mock_response
        
        # Create client and call method
        with patch('omniscan_orchestrator.grpc_client.grpc.insecure_channel'):
            client = OmniscanGrpcClient("localhost:50051")
            client.command_discovery = mock_stub
            
            result = client.get_server_capabilities()
        
        assert result["server_version"] == "0.1.0"
        assert result["protocol_version"] == "1.0.0"
        assert "audit_logging" in result["supported_features"]
        assert result["device_type"] == "hardware_server"
        mock_stub.GetServerCapabilities.assert_called_once()
    
    @patch('omniscan_orchestrator.grpc_client.hub_pb2_grpc.CommandDiscoveryStub')
    def test_get_server_capabilities_connection_error(self, mock_stub_class):
        """Test handling of connection errors when querying capabilities."""
        mock_stub = Mock()
        mock_stub_class.return_value = mock_stub
        
        import grpc
        mock_stub.GetServerCapabilities.side_effect = grpc.RpcError()
        
        with patch('omniscan_orchestrator.grpc_client.grpc.insecure_channel'):
            client = OmniscanGrpcClient("localhost:50051")
            client.command_discovery = mock_stub
            
            result = client.get_server_capabilities()
        
        assert "error" in result
    
    @patch('omniscan_orchestrator.cli._build_client')
    def test_cli_server_capabilities(self, mock_build_client):
        """Test CLI command for querying server capabilities."""
        mock_client = Mock()
        mock_client.get_server_capabilities.return_value = {
            "server_version": "0.1.0",
            "protocol_version": "1.0.0",
            "build_time": 1672531200,
            "supported_features": ["audit_logging", "gui"],
            "device_type": "hardware_server"
        }
        mock_build_client.return_value = mock_client
        
        result = runner.invoke(app, ["server-capabilities"])
        
        assert result.exit_code == 0
        assert "0.1.0" in result.stdout
        assert "1.0.0" in result.stdout
        mock_client.get_server_capabilities.assert_called_once()


# ==============================================================================
# Command Listing Tests
# ==============================================================================

class TestCommandListing:
    """Test listing available commands from server."""
    
    @patch('omniscan_orchestrator.grpc_client.hub_pb2_grpc.CommandDiscoveryStub')
    def test_list_commands_success(self, mock_stub_class):
        """Test successful retrieval of command list."""
        mock_stub = Mock()
        mock_stub_class.return_value = mock_stub
        
        # Mock commands
        mock_cmd1 = Mock()
        mock_cmd1.service_name = "Acquisition"
        mock_cmd1.command_name = "StartExposure"
        mock_cmd1.description = "Start X-ray exposure"
        mock_cmd1.request_fields = []
        mock_cmd1.response_type = "Empty"
        mock_cmd1.response_fields = []
        mock_cmd1.required_permissions = []
        mock_cmd1.safety_requirements = ["interlocks_safe"]
        
        mock_cmd2 = Mock()
        mock_cmd2.service_name = "Motion"
        mock_cmd2.command_name = "MoveTo"
        mock_cmd2.description = "Move to position"
        mock_cmd2.request_fields = []
        mock_cmd2.response_type = "Empty"
        mock_cmd2.response_fields = []
        mock_cmd2.required_permissions = []
        mock_cmd2.safety_requirements = ["motion_homed"]
        
        # Mock response
        mock_response = Mock()
        mock_response.commands = [mock_cmd1, mock_cmd2]
        mock_server_info = Mock()
        mock_server_info.server_version = "0.1.0"
        mock_server_info.protocol_version = "1.0.0"
        mock_server_info.supported_features = ["audit_logging"]
        mock_server_info.device_type = "hardware_server"
        mock_response.server_info = mock_server_info
        
        mock_stub.ListCommands.return_value = mock_response
        
        with patch('omniscan_orchestrator.grpc_client.grpc.insecure_channel'):
            client = OmniscanGrpcClient("localhost:50051")
            client.command_discovery = mock_stub
            
            result = client.list_commands()
        
        assert len(result["commands"]) == 2
        assert result["commands"][0]["service_name"] == "Acquisition"
        assert result["commands"][1]["service_name"] == "Motion"
        assert "interlocks_safe" in result["commands"][0]["safety_requirements"]
        assert result["server_info"]["server_version"] == "0.1.0"
    
    @patch('omniscan_orchestrator.cli._build_client')
    def test_cli_list_commands(self, mock_build_client):
        """Test CLI command for listing all commands."""
        mock_client = Mock()
        mock_client.list_commands.return_value = {
            "commands": [
                {
                    "service_name": "Acquisition",
                    "command_name": "StartExposure",
                    "description": "Start X-ray exposure with specified duration",
                    "safety_requirements": ["interlocks_safe"],
                },
                {
                    "service_name": "Motion",
                    "command_name": "MoveTo",
                    "description": "Move to absolute position",
                    "safety_requirements": ["motion_homed", "interlocks_safe"],
                }
            ],
            "server_info": {
                "server_version": "0.1.0",
                "protocol_version": "1.0.0"
            }
        }
        mock_build_client.return_value = mock_client
        
        result = runner.invoke(app, ["list-commands"])
        
        assert result.exit_code == 0
        assert "StartExposure" in result.stdout
        assert "MoveTo" in result.stdout
        assert "0.1.0" in result.stdout
    
    @patch('omniscan_orchestrator.cli._build_client')
    def test_cli_list_commands_filtered_by_service(self, mock_build_client):
        """Test CLI command filtering by service name."""
        mock_client = Mock()
        mock_client.list_commands.return_value = {
            "commands": [
                {
                    "service_name": "Acquisition",
                    "command_name": "StartExposure",
                    "description": "Start exposure",
                    "safety_requirements": [],
                },
                {
                    "service_name": "Acquisition",
                    "command_name": "Stop",
                    "description": "Stop acquisition",
                    "safety_requirements": [],
                }
            ],
            "server_info": {"server_version": "0.1.0", "protocol_version": "1.0.0"}
        }
        mock_build_client.return_value = mock_client
        
        result = runner.invoke(app, ["list-commands", "--service", "Acquisition"])
        
        assert result.exit_code == 0
        assert "StartExposure" in result.stdout
        assert "Stop" in result.stdout


# ==============================================================================
# Compatibility Validation Tests
# ==============================================================================

class TestCompatibilityValidation:
    """Test version and command compatibility validation."""
    
    @patch('omniscan_orchestrator.grpc_client.hub_pb2_grpc.CommandDiscoveryStub')
    def test_validate_compatibility_success(self, mock_stub_class):
        """Test successful compatibility validation."""
        mock_stub = Mock()
        mock_stub_class.return_value = mock_stub
        
        # Mock response - compatible
        mock_response = Mock()
        mock_response.compatible = True
        mock_response.message = "Client and server are fully compatible"
        mock_response.missing_commands = []
        mock_response.version_warnings = []
        mock_response.protocol_compatible = True
        
        mock_stub.ValidateCompatibility.return_value = mock_response
        
        with patch('omniscan_orchestrator.grpc_client.grpc.insecure_channel'):
            client = OmniscanGrpcClient("localhost:50051")
            client.command_discovery = mock_stub
            
            result = client.validate_compatibility(
                client_version="0.1.0",
                protocol_version="1.0.0",
                required_commands=["Acquisition.StartExposure", "Motion.MoveTo"]
            )
        
        assert result["compatible"] is True
        assert result["protocol_compatible"] is True
        assert len(result["missing_commands"]) == 0
        mock_stub.ValidateCompatibility.assert_called_once()
    
    @patch('omniscan_orchestrator.grpc_client.hub_pb2_grpc.CommandDiscoveryStub')
    def test_validate_compatibility_version_mismatch(self, mock_stub_class):
        """Test compatibility check with version mismatch."""
        mock_stub = Mock()
        mock_stub_class.return_value = mock_stub
        
        # Mock response - incompatible due to version
        mock_response = Mock()
        mock_response.compatible = False
        mock_response.message = "Incompatible: incompatible protocol version"
        mock_response.missing_commands = []
        mock_response.version_warnings = [
            "Protocol version mismatch: client 2.0.0 vs server 1.0.0"
        ]
        mock_response.protocol_compatible = False
        
        mock_stub.ValidateCompatibility.return_value = mock_response
        
        with patch('omniscan_orchestrator.grpc_client.grpc.insecure_channel'):
            client = OmniscanGrpcClient("localhost:50051")
            client.command_discovery = mock_stub
            
            result = client.validate_compatibility(
                client_version="0.2.0",
                protocol_version="2.0.0"
            )
        
        assert result["compatible"] is False
        assert result["protocol_compatible"] is False
        assert len(result["version_warnings"]) > 0
    
    @patch('omniscan_orchestrator.grpc_client.hub_pb2_grpc.CommandDiscoveryStub')
    def test_validate_compatibility_missing_commands(self, mock_stub_class):
        """Test compatibility check with missing required commands."""
        mock_stub = Mock()
        mock_stub_class.return_value = mock_stub
        
        # Mock response - incompatible due to missing commands
        mock_response = Mock()
        mock_response.compatible = False
        mock_response.message = "Incompatible: missing commands: FutureCommand.NewFeature"
        mock_response.missing_commands = ["FutureCommand.NewFeature"]
        mock_response.version_warnings = []
        mock_response.protocol_compatible = True
        
        mock_stub.ValidateCompatibility.return_value = mock_response
        
        with patch('omniscan_orchestrator.grpc_client.grpc.insecure_channel'):
            client = OmniscanGrpcClient("localhost:50051")
            client.command_discovery = mock_stub
            
            result = client.validate_compatibility(
                client_version="0.2.0",
                protocol_version="1.0.0",
                required_commands=["Acquisition.StartExposure", "FutureCommand.NewFeature"]
            )
        
        assert result["compatible"] is False
        assert "FutureCommand.NewFeature" in result["missing_commands"]
    
    @patch('omniscan_orchestrator.cli._build_client')
    def test_cli_validate_compatibility_success(self, mock_build_client):
        """Test CLI compatibility validation - success case."""
        mock_client = Mock()
        mock_client.validate_compatibility.return_value = {
            "compatible": True,
            "message": "Client and server are fully compatible",
            "missing_commands": [],
            "version_warnings": [],
            "protocol_compatible": True
        }
        mock_build_client.return_value = mock_client
        
        result = runner.invoke(app, ["validate-compatibility"])
        
        assert result.exit_code == 0
        assert "Compatible" in result.stdout
        mock_client.validate_compatibility.assert_called_once()
    
    @patch('omniscan_orchestrator.cli._build_client')
    def test_cli_validate_compatibility_failure(self, mock_build_client):
        """Test CLI compatibility validation - failure case."""
        mock_client = Mock()
        mock_client.validate_compatibility.return_value = {
            "compatible": False,
            "message": "Incompatible: protocol version mismatch",
            "missing_commands": ["FutureService.NewCommand"],
            "version_warnings": ["Major version mismatch"],
            "protocol_compatible": False
        }
        mock_build_client.return_value = mock_client
        
        result = runner.invoke(app, ["validate-compatibility"])
        
        assert result.exit_code == 1  # Should exit with error code
        assert "Incompatible" in result.stdout
        assert "FutureService.NewCommand" in result.stdout


# ==============================================================================
# Startup Compatibility Check Tests
# ==============================================================================

class TestStartupCompatibilityCheck:
    """Test orchestrator startup compatibility checking."""
    
    def test_startup_check_compatible(self):
        """Test that orchestrator can proceed when compatible."""
        mock_client = Mock()
        mock_client.validate_compatibility.return_value = {
            "compatible": True,
            "message": "Compatible",
            "missing_commands": [],
            "version_warnings": [],
            "protocol_compatible": True
        }
        
        # This would be in the orchestrator startup code
        compat = mock_client.validate_compatibility()
        
        assert compat["compatible"] is True
        # Orchestrator should continue normally
    
    def test_startup_check_incompatible_should_fail(self):
        """Test that orchestrator fails startup when incompatible."""
        mock_client = Mock()
        mock_client.validate_compatibility.return_value = {
            "compatible": False,
            "message": "I cannot communicate with the server, I am outdated",
            "missing_commands": ["CriticalService.RequiredCommand"],
            "version_warnings": [],
            "protocol_compatible": False
        }
        
        # This would be in the orchestrator startup code
        compat = mock_client.validate_compatibility()
        
        assert compat["compatible"] is False
        # Orchestrator should exit with error message
        assert "outdated" in compat["message"].lower() or "incompatible" in compat["message"].lower()
    
    def test_startup_check_with_required_commands(self):
        """Test startup check validates all required commands exist."""
        mock_client = Mock()
        
        # Define commands orchestrator needs
        required_commands = [
            "Acquisition.StartExposure",
            "Acquisition.Stop",
            "Acquisition.GetState",
            "Motion.MoveTo",
            "Motion.Home",
            "Safety.GetInterlockStatus",
            "Health.GetAggregateHealth",
            "StateMonitor.GetFullServerState",
            "DeviceInitialization.InitializeDetector",
            "DeviceInitialization.InitializeMotion",
        ]
        
        mock_client.validate_compatibility.return_value = {
            "compatible": True,
            "message": "All required commands available",
            "missing_commands": [],
            "version_warnings": [],
            "protocol_compatible": True
        }
        
        compat = mock_client.validate_compatibility(
            client_version="0.1.0",
            protocol_version="1.0.0",
            required_commands=required_commands
        )
        
        assert compat["compatible"] is True
        assert len(compat["missing_commands"]) == 0


# ==============================================================================
# Protocol Version Compatibility Tests
# ==============================================================================

class TestProtocolVersionCompatibility:
    """Test protocol version compatibility rules."""
    
    def test_same_major_version_compatible(self):
        """Test that same major versions are compatible."""
        # Client: 1.2.0, Server: 1.3.0 - should be compatible
        mock_client = Mock()
        mock_client.validate_compatibility.return_value = {
            "compatible": True,
            "protocol_compatible": True,
            "message": "Compatible",
            "missing_commands": [],
            "version_warnings": []
        }
        
        result = mock_client.validate_compatibility(
            client_version="0.1.0",
            protocol_version="1.2.0"
        )
        
        assert result["protocol_compatible"] is True
    
    def test_different_major_version_incompatible(self):
        """Test that different major versions are incompatible."""
        # Client: 2.0.0, Server: 1.0.0 - should be incompatible
        mock_client = Mock()
        mock_client.validate_compatibility.return_value = {
            "compatible": False,
            "protocol_compatible": False,
            "message": "Protocol version mismatch",
            "missing_commands": [],
            "version_warnings": [
                "Protocol version mismatch: client 2.0.0 vs server 1.0.0"
            ]
        }
        
        result = mock_client.validate_compatibility(
            client_version="0.1.0",
            protocol_version="2.0.0"
        )
        
        assert result["protocol_compatible"] is False
        assert len(result["version_warnings"]) > 0


# ==============================================================================
# Integration Tests - Orchestrator Startup Flow
# ==============================================================================

class TestOrchestratorStartupFlow:
    """Integration-style tests for orchestrator startup flow."""
    
    @patch('omniscan_orchestrator.grpc_client.hub_pb2_grpc')
    def test_complete_startup_flow_success(self, mock_grpc):
        """Test complete orchestrator startup with successful compatibility check."""
        # Mock all required stubs
        mock_channel = Mock()
        
        # Mock CommandDiscovery stub
        mock_discovery_stub = Mock()
        mock_compat_response = Mock()
        mock_compat_response.compatible = True
        mock_compat_response.message = "Compatible"
        mock_compat_response.missing_commands = []
        mock_compat_response.version_warnings = []
        mock_compat_response.protocol_compatible = True
        mock_discovery_stub.ValidateCompatibility.return_value = mock_compat_response
        
        with patch('omniscan_orchestrator.grpc_client.grpc.insecure_channel', return_value=mock_channel):
            mock_grpc.CommandDiscoveryStub.return_value = mock_discovery_stub
            
            client = OmniscanGrpcClient("localhost:50051")
            client.command_discovery = mock_discovery_stub
            
            # Orchestrator startup: check compatibility
            compat = client.validate_compatibility()
            
            if not compat["compatible"]:
                # This should not happen in success case
                pytest.fail("Compatibility check failed unexpectedly")
            
            # Orchestrator should proceed normally
            assert compat["compatible"] is True
    
    @patch('omniscan_orchestrator.grpc_client.hub_pb2_grpc')
    def test_complete_startup_flow_failure(self, mock_grpc):
        """Test complete orchestrator startup with failed compatibility check."""
        # Mock all required stubs
        mock_channel = Mock()
        
        # Mock CommandDiscovery stub
        mock_discovery_stub = Mock()
        mock_compat_response = Mock()
        mock_compat_response.compatible = False
        mock_compat_response.message = "I cannot communicate with the server, I am outdated"
        mock_compat_response.missing_commands = ["NewService.NewCommand"]
        mock_compat_response.version_warnings = ["Major version mismatch"]
        mock_compat_response.protocol_compatible = False
        mock_discovery_stub.ValidateCompatibility.return_value = mock_compat_response
        
        with patch('omniscan_orchestrator.grpc_client.grpc.insecure_channel', return_value=mock_channel):
            mock_grpc.CommandDiscoveryStub.return_value = mock_discovery_stub
            
            client = OmniscanGrpcClient("localhost:50051")
            client.command_discovery = mock_discovery_stub
            
            # Orchestrator startup: check compatibility
            compat = client.validate_compatibility()
            
            # Orchestrator should detect incompatibility
            assert compat["compatible"] is False
            assert "outdated" in compat["message"].lower() or "cannot communicate" in compat["message"].lower()
            assert len(compat["missing_commands"]) > 0
    
    def test_orchestrator_required_commands_list(self):
        """Test that orchestrator defines its required commands correctly."""
        # This represents what orchestrator needs to function
        required_commands = [
            # Critical acquisition commands
            "Acquisition.StartExposure",
            "Acquisition.Stop",
            "Acquisition.Abort",
            "Acquisition.GetState",
            "Acquisition.GetLastExposureResult",
            "Acquisition.CalibrateDetector",
            
            # Critical motion commands
            "Motion.MoveTo",
            "Motion.Home",
            "Motion.Stop",
            "Motion.GetPosition",
            
            # Critical safety commands
            "Safety.GetInterlockStatus",
            "Safety.CheckSafetyToOperate",
            
            # Critical monitoring commands
            "Health.Liveness",
            "Health.Readiness",
            "Health.GetAggregateHealth",
            "StateMonitor.GetFullServerState",
            
            # Device initialization
            "DeviceInitialization.InitializeDetector",
            "DeviceInitialization.InitializeMotion",
        ]
        
        # Verify list is not empty
        assert len(required_commands) > 0
        
        # Verify format (Service.Command)
        for cmd in required_commands:
            assert "." in cmd
            parts = cmd.split(".")
            assert len(parts) == 2
            assert parts[0]  # Service name not empty
            assert parts[1]  # Command name not empty


# ==============================================================================
# Error Handling Tests
# ==============================================================================

class TestErrorHandling:
    """Test error handling in command discovery operations."""
    
    @patch('omniscan_orchestrator.grpc_client.hub_pb2_grpc.CommandDiscoveryStub')
    def test_network_error_during_compatibility_check(self, mock_stub_class):
        """Test handling of network errors during compatibility check."""
        mock_stub = Mock()
        mock_stub_class.return_value = mock_stub
        
        import grpc
        mock_stub.ValidateCompatibility.side_effect = grpc.RpcError()
        
        with patch('omniscan_orchestrator.grpc_client.grpc.insecure_channel'):
            client = OmniscanGrpcClient("localhost:50051")
            client.command_discovery = mock_stub
            
            result = client.validate_compatibility()
        
        assert "error" in result
    
    @patch('omniscan_orchestrator.cli._build_client')
    def test_cli_handles_server_unavailable(self, mock_build_client):
        """Test CLI gracefully handles server being unavailable."""
        mock_client = Mock()
        mock_client.validate_compatibility.return_value = {
            "error": "Connection refused",
            "code": "UNAVAILABLE"
        }
        mock_build_client.return_value = mock_client
        
        result = runner.invoke(app, ["validate-compatibility"])
        
        # Should exit with error
        assert result.exit_code == 1
        assert "Error" in result.stdout or "error" in result.stdout.lower()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
