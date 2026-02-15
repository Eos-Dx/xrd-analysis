"""
Unit Tests for CLI Orchestrator
Medical Device Software - FDA Compliant Testing

Tests cover:
- Client building with various authentication methods
- Status and health commands
- Maintenance operations
- Configuration management
- Device control workflow
- Certificate management
- USB stub operations
"""

import pytest
import json
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock, call
from typer.testing import CliRunner
from datetime import datetime

from omniscan_orchestrator.cli import app, _build_client
from omniscan_orchestrator.grpc_client import OmniscanGrpcClient


runner = CliRunner()


# ==============================================================================
# Client Building Tests
# ==============================================================================

class TestClientBuilding:
    """Test the _build_client function with various authentication methods."""
    
    @patch('omniscan_orchestrator.cli.OmniscanGrpcClient')
    def test_build_client_with_cert_and_key(self, mock_client_class):
        """Test client building with certificate and key files."""
        server = "localhost:50051"
        cert = Path("/path/to/cert.pem")
        key = Path("/path/to/key.pem")
        ca_cert = Path("/path/to/ca.pem")
        
        _build_client(server, None, cert, key, ca_cert)
        
        mock_client_class.assert_called_once_with(
            server_address=server,
            client_cert_path=str(cert),
            client_key_path=str(key),
            ca_cert_path=str(ca_cert)
        )
    
    @patch('omniscan_orchestrator.cli.MaintenanceUSBStub')
    @patch('omniscan_orchestrator.cli.OmniscanGrpcClient')
    def test_build_client_with_usb_stub(self, mock_client_class, mock_usb_stub_class):
        """Test client building with USB stub."""
        server = "localhost:50051"
        usb_path = Path("/path/to/usb")
        
        mock_usb_instance = Mock()
        mock_identity = Mock()
        mock_identity.fingerprint_sha256 = "abc123"
        mock_usb_instance.load.return_value = mock_identity
        mock_usb_stub_class.return_value = mock_usb_instance
        
        _build_client(server, usb_path, None, None, None)
        
        mock_usb_stub_class.assert_called_once_with(usb_path)
        mock_client_class.assert_called_once()
    
    @patch('omniscan_orchestrator.cli.OmniscanGrpcClient')
    def test_build_client_insecure(self, mock_client_class):
        """Test client building without any authentication (insecure)."""
        server = "localhost:50051"
        
        _build_client(server, None, None, None, None)
        
        mock_client_class.assert_called_once_with(
            server_address=server,
            client_cert_path=None,
            client_key_path=None,
            ca_cert_path=None
        )


# ==============================================================================
# Status and Health Command Tests
# ==============================================================================

class TestStatusAndHealthCommands:
    """Test status and health monitoring commands."""
    
    @patch('omniscan_orchestrator.cli._build_client')
    def test_status_command_success(self, mock_build_client):
        """Test status command returns server status."""
        mock_client = Mock()
        mock_client.get_status.return_value = {
            "ok": True,
            "state": "IDLE",
            "interlocks": {"overall_safe": True}
        }
        mock_build_client.return_value = mock_client
        
        result = runner.invoke(app, ["status"])
        
        assert result.exit_code == 0
        assert "ok" in result.stdout
        mock_client.get_status.assert_called_once()
    
    @patch('omniscan_orchestrator.cli._build_client')
    def test_status_command_with_cert_options(self, mock_build_client):
        """Test status command with certificate authentication."""
        mock_client = Mock()
        mock_client.get_status.return_value = {"ok": True}
        mock_build_client.return_value = mock_client
        
        with tempfile.NamedTemporaryFile(delete=False) as cert_file, \
             tempfile.NamedTemporaryFile(delete=False) as key_file, \
             tempfile.NamedTemporaryFile(delete=False) as ca_file:
            
            result = runner.invoke(app, [
                "status",
                "--cert", cert_file.name,
                "--key", key_file.name,
                "--ca-cert", ca_file.name
            ])
            
            assert result.exit_code == 0
            mock_build_client.assert_called_once()
    
    @patch('omniscan_orchestrator.cli._build_client')
    def test_health_command(self, mock_build_client):
        """Test health command returns aggregate health."""
        mock_client = Mock()
        mock_client.get_aggregate_health.return_value = {
            "ok": True,
            "components": [
                {"name": "Detector", "ok": True, "detail": "Operational"}
            ],
            "interlocks": {"overall_safe": True}
        }
        mock_build_client.return_value = mock_client
        
        result = runner.invoke(app, ["health"])
        
        assert result.exit_code == 0
        assert "ok" in result.stdout
        assert "components" in result.stdout
        mock_client.get_aggregate_health.assert_called_once()
    
    @patch('omniscan_orchestrator.cli._build_client')
    def test_interlocks_command(self, mock_build_client):
        """Test interlocks command returns safety status."""
        mock_client = Mock()
        mock_client.get_interlocks.return_value = {
            "overall_safe": True,
            "emergency_stop": False,
            "door_closed": True,
            "radiation_safe": True,
            "cooling_ok": True,
            "power_ok": True
        }
        mock_build_client.return_value = mock_client
        
        result = runner.invoke(app, ["interlocks"])
        
        assert result.exit_code == 0
        assert "overall_safe" in result.stdout
        mock_client.get_interlocks.assert_called_once()
    
    @patch('omniscan_orchestrator.cli._build_client')
    def test_state_command(self, mock_build_client):
        """Test state command returns server state."""
        mock_client = Mock()
        mock_client.get_server_state.return_value = {
            "state": "IDLE",
            "detail": "System ready"
        }
        mock_build_client.return_value = mock_client
        
        result = runner.invoke(app, ["state"])
        
        assert result.exit_code == 0
        assert "state" in result.stdout
        mock_client.get_server_state.assert_called_once()
    
    @patch('omniscan_orchestrator.cli._build_client')
    def test_detector_health_command(self, mock_build_client):
        """Test detector-health command."""
        mock_client = Mock()
        mock_client.get_detector_health.return_value = {
            "powered": True,
            "temperature": 25.5,
            "status": "READY"
        }
        mock_build_client.return_value = mock_client
        
        result = runner.invoke(app, ["detector-health"])
        
        assert result.exit_code == 0
        assert "powered" in result.stdout
        mock_client.get_detector_health.assert_called_once()
    
    @patch('omniscan_orchestrator.cli._build_client')
    def test_motion_health_command(self, mock_build_client):
        """Test motion-health command."""
        mock_client = Mock()
        mock_client.get_motion_health.return_value = {
            "status": "OK",
            "position": 0.0
        }
        mock_build_client.return_value = mock_client
        
        result = runner.invoke(app, ["motion-health"])
        
        assert result.exit_code == 0
        mock_client.get_motion_health.assert_called_once()


# ==============================================================================
# Maintenance Operations Tests
# ==============================================================================

class TestMaintenanceOperations:
    """Test maintenance mode operations."""
    
    @patch('omniscan_orchestrator.cli._build_client')
    def test_enter_maintenance(self, mock_build_client):
        """Test entering maintenance mode."""
        mock_client = Mock()
        mock_client.enter_maintenance.return_value = {
            "success": True,
            "lease_id": "lease-123",
            "ttl": 900
        }
        mock_build_client.return_value = mock_client
        
        result = runner.invoke(app, ["enter-maintenance", "--ttl", "900"])
        
        assert result.exit_code == 0
        mock_client.enter_maintenance.assert_called_once_with(900)
    
    @patch('omniscan_orchestrator.cli._build_client')
    def test_renew_maintenance(self, mock_build_client):
        """Test renewing maintenance lease."""
        mock_client = Mock()
        mock_client.renew_maintenance.return_value = {
            "success": True,
            "new_ttl": 900
        }
        mock_build_client.return_value = mock_client
        
        result = runner.invoke(app, ["renew", "--ttl", "900"])
        
        assert result.exit_code == 0
        mock_client.renew_maintenance.assert_called_once_with(900)
    
    @patch('omniscan_orchestrator.cli._build_client')
    def test_exit_maintenance(self, mock_build_client):
        """Test exiting maintenance mode."""
        mock_client = Mock()
        mock_client.exit_maintenance.return_value = {
            "success": True
        }
        mock_build_client.return_value = mock_client
        
        result = runner.invoke(app, ["exit-maintenance"])
        
        assert result.exit_code == 0
        mock_client.exit_maintenance.assert_called_once()


# ==============================================================================
# Configuration Management Tests
# ==============================================================================

class TestConfigurationManagement:
    """Test configuration management commands."""
    
    @patch('omniscan_orchestrator.cli._build_client')
    def test_get_config(self, mock_build_client):
        """Test retrieving configuration."""
        mock_client = Mock()
        mock_client.get_config.return_value = {
            "detector": {"exposure_time": 1000},
            "safety": {"max_dose": 100}
        }
        mock_build_client.return_value = mock_client
        
        result = runner.invoke(app, ["get-config"])
        
        assert result.exit_code == 0
        assert "detector" in result.stdout
        mock_client.get_config.assert_called_once()
    
    @patch('omniscan_orchestrator.cli._build_client')
    def test_set_config(self, mock_build_client):
        """Test setting configuration from file."""
        mock_client = Mock()
        mock_client.set_config.return_value = {"success": True}
        mock_build_client.return_value = mock_client
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            config = {"detector": {"exposure_time": 2000}}
            json.dump(config, f)
            f.flush()
            
            result = runner.invoke(app, ["set-config", "--file", f.name])
            
            assert result.exit_code == 0
            mock_client.set_config.assert_called_once()
            # Verify the config data was passed
            call_args = mock_client.set_config.call_args[0][0]
            assert call_args["detector"]["exposure_time"] == 2000
    
    @patch('omniscan_orchestrator.cli._build_client')
    def test_patch_config(self, mock_build_client):
        """Test patching configuration with inline JSON."""
        mock_client = Mock()
        mock_client.patch_config.return_value = {"success": True}
        mock_build_client.return_value = mock_client
        
        patch_data = '{"detector": {"exposure_time": 1500}}'
        result = runner.invoke(app, ["patch-config", "--patch", patch_data])
        
        assert result.exit_code == 0
        mock_client.patch_config.assert_called_once()
    
    @patch('omniscan_orchestrator.cli._build_client')
    def test_validate_config(self, mock_build_client):
        """Test validating configuration."""
        mock_client = Mock()
        mock_client.validate_config.return_value = {
            "valid": True,
            "errors": []
        }
        mock_build_client.return_value = mock_client
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            config = {"detector": {"exposure_time": 1000}}
            json.dump(config, f)
            f.flush()
            
            result = runner.invoke(app, ["validate-config", "--file", f.name])
            
            assert result.exit_code == 0
            mock_client.validate_config.assert_called_once()


# ==============================================================================
# Device Control Tests
# ==============================================================================

class TestDeviceControl:
    """Test device control commands."""
    
    @patch('requests.get')
    def test_device_state(self, mock_get):
        """Test getting device state."""
        mock_response = Mock()
        mock_response.json.return_value = {"state": "IDLE"}
        mock_get.return_value = mock_response
        
        result = runner.invoke(app, ["device-state"])
        
        assert result.exit_code == 0
        mock_get.assert_called_once()
    
    @patch('requests.post')
    def test_device_power_on(self, mock_post):
        """Test turning device power on."""
        mock_response = Mock()
        mock_response.json.return_value = {"success": True}
        mock_post.return_value = mock_response
        
        result = runner.invoke(app, ["device-power", "--on"])
        
        assert result.exit_code == 0
        mock_post.assert_called_once()
        call_args = mock_post.call_args
        assert call_args[1]['json']['on'] is True
    
    @patch('requests.post')
    def test_device_power_off(self, mock_post):
        """Test turning device power off."""
        mock_response = Mock()
        mock_response.json.return_value = {"success": True}
        mock_post.return_value = mock_response
        
        result = runner.invoke(app, ["device-power", "--off"])
        
        assert result.exit_code == 0
        call_args = mock_post.call_args
        assert call_args[1]['json']['on'] is False
    
    @patch('requests.post')
    def test_measure_start(self, mock_post):
        """Test starting a measurement."""
        mock_response = Mock()
        mock_response.json.return_value = {"measurement_id": "m-123"}
        mock_post.return_value = mock_response
        
        result = runner.invoke(app, [
            "measure-start",
            "--duration", "60",
            "--mode", "calibrant"
        ])
        
        assert result.exit_code == 0
        mock_post.assert_called_once()
        call_args = mock_post.call_args
        assert call_args[1]['json']['duration_s'] == 60
        assert call_args[1]['json']['mode'] == "calibrant"
    
    @patch('requests.post')
    def test_measure_stop(self, mock_post):
        """Test stopping a measurement."""
        mock_response = Mock()
        mock_response.json.return_value = {"success": True}
        mock_post.return_value = mock_response
        
        result = runner.invoke(app, ["measure-stop"])
        
        assert result.exit_code == 0
        mock_post.assert_called_once()
    
    @patch('requests.get')
    def test_measure_status(self, mock_get):
        """Test getting measurement status."""
        mock_response = Mock()
        mock_response.json.return_value = {
            "status": "running",
            "progress": 50
        }
        mock_get.return_value = mock_response
        
        result = runner.invoke(app, ["measure-status"])
        
        assert result.exit_code == 0
        mock_get.assert_called_once()
    
    @patch('requests.get')
    def test_measure_result(self, mock_get):
        """Test getting measurement result."""
        mock_response = Mock()
        mock_response.json.return_value = {
            "has_result": True,
            "data_size": 1024
        }
        mock_get.return_value = mock_response
        
        result = runner.invoke(app, ["measure-result"])
        
        assert result.exit_code == 0
        mock_get.assert_called_once()


# ==============================================================================
# Calibration Tests
# ==============================================================================

class TestCalibration:
    """Test detector calibration command."""
    
    @patch('omniscan_orchestrator.cli._build_client')
    def test_calibrate_detector_success(self, mock_build_client):
        """Test successful detector calibration."""
        mock_client = Mock()
        mock_client.calibrate_detector.return_value = {
            "success": True,
            "calibration_id": "cal-123"
        }
        mock_client.close = Mock()
        mock_build_client.return_value = mock_client
        
        result = runner.invoke(app, [
            "calibrate-detector",
            "--user", "operator:OP001"
        ])
        
        assert result.exit_code == 0
        assert "successfully" in result.stdout.lower()
        mock_client.calibrate_detector.assert_called_once_with("operator:OP001")
        mock_client.close.assert_called_once()
    
    @patch('omniscan_orchestrator.cli._build_client')
    def test_calibrate_detector_failure(self, mock_build_client):
        """Test failed detector calibration."""
        mock_client = Mock()
        mock_client.calibrate_detector.return_value = {
            "error": "Calibration failed: detector not ready"
        }
        mock_client.close = Mock()
        mock_build_client.return_value = mock_client
        
        result = runner.invoke(app, ["calibrate-detector"])
        
        assert result.exit_code == 1
        assert "failed" in result.stdout.lower()
        mock_client.close.assert_called_once()


# ==============================================================================
# Measurement Workflow Tests
# ==============================================================================

class TestMeasurementWorkflow:
    """Test the complete measurement workflow command."""
    
    @patch('omniscan_orchestrator.cli._build_client')
    @patch('time.sleep')  # Speed up tests
    def test_workflow_success(self, mock_sleep, mock_build_client):
        """Test successful complete measurement workflow."""
        mock_client = Mock()
        
        # Mock successful responses for all workflow steps
        mock_client.get_status.return_value = {
            "state": "IDLE",
            "interlocks": {"overall_safe": True},
            "components": []
        }
        mock_client.get_device_state.return_value = {
            "detector": {"powered": True, "status": "READY"}
        }
        mock_client.start_measurement.return_value = {
            "success": True,
            "measurement_id": "m-123"
        }
        mock_client.get_measurement_result.return_value = {
            "has_result": True,
            "exposure_time_ms": 60000,
            "data_size": 2048,
            "detector_temp": 25.0
        }
        mock_client.close = Mock()
        mock_build_client.return_value = mock_client
        
        result = runner.invoke(app, [
            "measure-workflow",
            "--duration", "1",  # Short duration for testing
            "--user", "operator:OP001"
        ])
        
        assert result.exit_code == 0
        assert "completed" in result.stdout.lower()
        mock_client.close.assert_called_once()
    
    @patch('omniscan_orchestrator.cli._build_client')
    def test_workflow_interlock_failure(self, mock_build_client):
        """Test workflow failure due to interlocks."""
        mock_client = Mock()
        mock_client.get_status.return_value = {
            "state": "IDLE",
            "interlocks": {
                "overall_safe": False,
                "emergency_stop": True,
                "door_closed": False
            },
            "components": []
        }
        mock_client.close = Mock()
        mock_build_client.return_value = mock_client
        
        result = runner.invoke(app, ["measure-workflow"])
        
        assert result.exit_code == 1
        assert "interlock" in result.stdout.lower()
        mock_client.close.assert_called_once()
    
    @patch('omniscan_orchestrator.cli._build_client')
    @patch('time.sleep')
    def test_workflow_with_calibration(self, mock_sleep, mock_build_client):
        """Test workflow that requires calibration."""
        mock_client = Mock()
        
        # System locked, needs calibration
        mock_client.get_status.return_value = {
            "state": "SAFE",
            "interlocks": {"overall_safe": True},
            "components": [
                {"name": "Safety State Machine", "ok": True, "detail": "LOCKED"}
            ]
        }
        mock_client.calibrate_detector.return_value = {"success": True}
        mock_client.get_device_state.return_value = {
            "detector": {"powered": True, "status": "READY"}
        }
        mock_client.start_measurement.return_value = {"success": True}
        mock_client.get_measurement_result.return_value = {
            "has_result": True,
            "exposure_time_ms": 1000,
            "data_size": 1024
        }
        mock_client.close = Mock()
        mock_build_client.return_value = mock_client
        
        result = runner.invoke(app, ["measure-workflow", "--duration", "1"])
        
        assert result.exit_code == 0
        mock_client.calibrate_detector.assert_called_once()
    
    @patch('omniscan_orchestrator.cli._build_client')
    @patch('time.sleep')
    def test_workflow_power_on_detector(self, mock_sleep, mock_build_client):
        """Test workflow that powers on detector."""
        mock_client = Mock()
        
        mock_client.get_status.return_value = {
            "state": "IDLE",
            "interlocks": {"overall_safe": True},
            "components": []
        }
        # Detector initially off
        mock_client.get_device_state.return_value = {
            "detector": {"powered": False, "status": "OFF"}
        }
        mock_client.power_device.return_value = {"success": True}
        mock_client.start_measurement.return_value = {"success": True}
        mock_client.get_measurement_result.return_value = {
            "has_result": True,
            "exposure_time_ms": 1000
        }
        mock_client.close = Mock()
        mock_build_client.return_value = mock_client
        
        result = runner.invoke(app, ["measure-workflow", "--duration", "1"])
        
        assert result.exit_code == 0
        mock_client.power_device.assert_called_once_with("detector", True, "operator")


# ==============================================================================
# Certificate Management Tests
# ==============================================================================

class TestCertificateManagement:
    """Test certificate management commands."""
    
    @patch('omniscan_orchestrator.cli.CertificateManager')
    def test_cert_generate(self, mock_cert_manager_class):
        """Test generating an engineer certificate."""
        mock_manager = Mock()
        mock_cert_manager_class.return_value = mock_manager
        
        cert_path = Path("/path/to/cert.crt")
        key_path = Path("/path/to/key.key")
        mock_manager.create_engineer_certificate.return_value = (cert_path, key_path)
        mock_manager.get_certificate_info.return_value = {
            "common_name": "engineer:ENG001",
            "san_uris": ["urn:omniscan:device:ABC123"],
            "not_valid_before": datetime(2025, 1, 1),
            "not_valid_after": datetime(2025, 1, 2),
            "is_valid": True
        }
        
        result = runner.invoke(app, [
            "cert", "generate",
            "--engineer-id", "ENG001",
            "--device-uuid", "ABC123",
            "--validity-days", "1"
        ])
        
        assert result.exit_code == 0
        assert "successfully" in result.stdout.lower()
        assert "ENG001" in result.stdout
        mock_manager.create_engineer_certificate.assert_called_once()
    
    @patch('omniscan_orchestrator.cli.CertificateManager')
    def test_cert_list(self, mock_cert_manager_class):
        """Test listing engineer certificates."""
        mock_manager = Mock()
        mock_cert_manager_class.return_value = mock_manager
        
        mock_manager.list_engineer_certificates.return_value = [
            {
                "common_name": "engineer:ENG001",
                "is_valid": True,
                "not_valid_before": datetime(2025, 1, 1),
                "not_valid_after": datetime(2025, 1, 2),
                "san_uris": ["urn:omniscan:device:ABC123"],
                "file_path": "/path/to/cert.crt"
            }
        ]
        
        result = runner.invoke(app, ["cert", "list"])
        
        assert result.exit_code == 0
        assert "ENG001" in result.stdout
        mock_manager.list_engineer_certificates.assert_called_once()
    
    @patch('omniscan_orchestrator.cli.CertificateManager')
    def test_cert_list_empty(self, mock_cert_manager_class):
        """Test listing certificates when none exist."""
        mock_manager = Mock()
        mock_cert_manager_class.return_value = mock_manager
        mock_manager.list_engineer_certificates.return_value = []
        
        result = runner.invoke(app, ["cert", "list"])
        
        assert result.exit_code == 0
        assert "No engineer certificates" in result.stdout
    
    @patch('omniscan_orchestrator.cli.CertificateManager')
    def test_cert_info(self, mock_cert_manager_class):
        """Test showing certificate information."""
        mock_manager = Mock()
        mock_cert_manager_class.return_value = mock_manager
        
        mock_manager.get_certificate_info.return_value = {
            "common_name": "engineer:ENG001",
            "subject": "CN=engineer:ENG001,O=Omniscan",
            "issuer": "CN=Maintenance CA,O=Omniscan",
            "serial_number": 12345,
            "not_valid_before": datetime(2025, 1, 1),
            "not_valid_after": datetime(2025, 1, 2),
            "is_valid": True,
            "san_uris": ["urn:omniscan:device:ABC123"]
        }
        
        with tempfile.NamedTemporaryFile(suffix='.crt', delete=False) as f:
            result = runner.invoke(app, ["cert", "info", "--cert-path", f.name])
        
        assert result.exit_code == 0
        assert "engineer:ENG001" in result.stdout


# ==============================================================================
# USB Stub Tests
# ==============================================================================

class TestUSBStub:
    """Test USB stub development utilities."""
    
    @patch('omniscan_orchestrator.cli.MaintenanceUSBStub')
    def test_usb_dev_init(self, mock_usb_stub_class):
        """Test initializing a USB dev stub."""
        mock_stub = Mock()
        mock_identity = Mock()
        mock_identity.fingerprint_sha256 = "abc123def456"
        mock_identity.metadata = {
            "type": "dev-maintenance-usb-stub",
            "server_uuid": "SRV-001"
        }
        mock_stub.load.return_value = mock_identity
        mock_usb_stub_class.create_dev_stub.return_value = mock_stub
        
        with tempfile.TemporaryDirectory() as temp_dir:
            result = runner.invoke(app, [
                "usb-dev", "init",
                "--path", temp_dir,
                "--server-uuid", "SRV-001",
                "--common-name", "test-usb",
                "--days", "30"
            ])
        
        assert result.exit_code == 0
        assert "abc123def456" in result.stdout
        mock_usb_stub_class.create_dev_stub.assert_called_once()


# ==============================================================================
# Interactive Session Tests
# ==============================================================================

class TestInteractiveSession:
    """Test interactive session command."""
    
    @patch('omniscan_orchestrator.cli.InteractiveSession')
    @patch('omniscan_orchestrator.cli._build_client')
    def test_interactive_command(self, mock_build_client, mock_session_class):
        """Test starting an interactive session."""
        mock_client = Mock()
        mock_build_client.return_value = mock_client
        
        mock_session = Mock()
        mock_session.run.return_value = None
        mock_session_class.return_value = mock_session
        
        result = runner.invoke(app, ["interactive"], input="\n")
        
        # Interactive session may hang, so we just check it was called
        mock_session_class.assert_called_once()
        mock_session.run.assert_called_once()


# ==============================================================================
# Error Handling Tests
# ==============================================================================

class TestErrorHandling:
    """Test error handling in CLI commands."""
    
    @patch('omniscan_orchestrator.cli._build_client')
    def test_status_command_connection_error(self, mock_build_client):
        """Test status command handles connection errors."""
        mock_client = Mock()
        mock_client.get_status.return_value = {
            "error": "Connection refused",
            "code": "UNAVAILABLE"
        }
        mock_build_client.return_value = mock_client
        
        result = runner.invoke(app, ["status"])
        
        assert result.exit_code == 0  # Command runs but shows error
        assert "error" in result.stdout.lower()
    
    def test_set_config_file_not_found(self):
        """Test set-config with non-existent file."""
        result = runner.invoke(app, [
            "set-config",
            "--file", "/nonexistent/file.json"
        ])
        
        # Typer validates file existence
        assert result.exit_code != 0
    
    @patch('omniscan_orchestrator.cli.CertificateManager')
    def test_cert_generate_failure(self, mock_cert_manager_class):
        """Test certificate generation failure."""
        mock_manager = Mock()
        mock_cert_manager_class.return_value = mock_manager
        mock_manager.create_engineer_certificate.side_effect = RuntimeError("CA not found")
        
        result = runner.invoke(app, [
            "cert", "generate",
            "--engineer-id", "ENG001",
            "--device-uuid", "ABC123"
        ])
        
        assert result.exit_code == 1
        # Error message is in stderr or stdout depending on exception handling
        output = result.stdout + (result.stderr or "")
        assert "Error" in output or "error" in output.lower()


# ==============================================================================
# Integration-Style Tests
# ==============================================================================

class TestCLIIntegration:
    """Integration-style tests for CLI command combinations."""
    
    @patch('omniscan_orchestrator.cli._build_client')
    def test_full_maintenance_workflow(self, mock_build_client):
        """Test complete maintenance workflow: enter, check status, exit."""
        mock_client = Mock()
        mock_client.enter_maintenance.return_value = {"success": True}
        mock_client.get_status.return_value = {"ok": True, "state": "MAINTENANCE"}
        mock_client.exit_maintenance.return_value = {"success": True}
        mock_build_client.return_value = mock_client
        
        # Enter maintenance
        result1 = runner.invoke(app, ["enter-maintenance"])
        assert result1.exit_code == 0
        
        # Check status
        result2 = runner.invoke(app, ["status"])
        assert result2.exit_code == 0
        
        # Exit maintenance
        result3 = runner.invoke(app, ["exit-maintenance"])
        assert result3.exit_code == 0
    
    @patch('omniscan_orchestrator.cli._build_client')
    def test_config_lifecycle(self, mock_build_client):
        """Test config get, validate, set workflow."""
        mock_client = Mock()
        original_config = {"detector": {"exposure_time": 1000}}
        mock_client.get_config.return_value = original_config
        mock_client.validate_config.return_value = {"valid": True}
        mock_client.set_config.return_value = {"success": True}
        mock_build_client.return_value = mock_client
        
        # Get current config
        result1 = runner.invoke(app, ["get-config"])
        assert result1.exit_code == 0
        
        # Validate new config
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            new_config = {"detector": {"exposure_time": 2000}}
            json.dump(new_config, f)
            f.flush()
            
            result2 = runner.invoke(app, ["validate-config", "--file", f.name])
            assert result2.exit_code == 0
            
            # Set new config
            result3 = runner.invoke(app, ["set-config", "--file", f.name])
            assert result3.exit_code == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
