"""
Unit tests for MarlinStageController.

These tests use mocking to avoid requiring actual hardware.
"""

import pytest
import time
import queue
from unittest.mock import Mock, MagicMock, patch, call
import sys
from pathlib import Path

# Add hardware directory to path
hardware_path = Path(__file__).resolve().parents[1] / "hardware"
sys.path.insert(0, str(hardware_path))

legacy_stage_source = hardware_path / "difra" / "hardware" / "xystages.py"
if not legacy_stage_source.is_file():
    pytest.skip(
        "legacy Marlin stage subsystem is not present in this checkout",
        allow_module_level=True,
    )

# Mock serial module before importing xystages
sys.modules['serial'] = MagicMock()
sys.modules['serial.tools'] = MagicMock()
sys.modules['serial.tools.list_ports'] = MagicMock()

from xystages import MarlinStageController, StageAxisLimitError, BaseStageController


class TestMarlinStageControllerInit:
    """Test MarlinStageController initialization."""

    def test_init_with_minimal_config(self):
        """Test initialization with minimal configuration."""
        config = {
            "id": "COM4",
            "alias": "TEST_STAGE"
        }
        
        stage = MarlinStageController(config)
        
        assert stage.alias == "TEST_STAGE"
        assert stage.port == "COM4"
        assert stage.baudrate == 115200
        assert stage.feedrate == 3000
        assert stage._x == 0.0
        assert stage._y == 0.0

    def test_init_with_full_config(self):
        """Test initialization with full configuration."""
        config = {
            "id": "COM5",
            "alias": "MOLI_STAGE",
            "settings": {
                "limits_mm": {
                    "x": {"min": 0.0, "max": 90.0},
                    "y": {"min": 0.0, "max": 100.0}
                },
                "home": [0.0, 0.0],
                "load": [90.0, 0.0]
            }
        }
        
        stage = MarlinStageController(
            config,
            baudrate=250000,
            feedrate=5000,
            homing_timeout=20
        )
        
        assert stage.alias == "MOLI_STAGE"
        assert stage.port == "COM5"
        assert stage.baudrate == 250000
        assert stage.feedrate == 5000
        assert stage.homing_timeout == 20
        
        limits = stage.get_limits()
        assert limits["x"] == (0.0, 90.0)
        assert limits["y"] == (0.0, 100.0)
        
        positions = stage.get_home_load_positions()
        assert positions["home"] == (0.0, 0.0)
        assert positions["load"] == (90.0, 0.0)

    def test_inherits_from_base_controller(self):
        """Test that MarlinStageController inherits from BaseStageController."""
        config = {"id": "COM4", "alias": "TEST"}
        stage = MarlinStageController(config)
        
        assert isinstance(stage, BaseStageController)

    def test_default_limits(self):
        """Test default limits when not specified in config."""
        config = {"id": "COM4", "alias": "TEST"}
        stage = MarlinStageController(config)
        
        limits = stage.get_limits()
        assert limits["x"] == (0.0, 90.0)
        assert limits["y"] == (0.0, 100.0)


class TestMarlinStageControllerConnection:
    """Test serial connection management."""

    @patch('serial.Serial')
    @patch('threading.Thread')
    def test_init_stage_success(self, mock_thread, mock_serial):
        """Test successful stage initialization."""
        config = {"id": "COM4", "alias": "TEST"}
        stage = MarlinStageController(config)
        
        # Mock serial connection
        mock_ser = MagicMock()
        mock_serial.return_value = mock_ser
        
        result = stage.init_stage()
        
        assert result is True
        mock_serial.assert_called_once_with("COM4", 115200, timeout=1.0)
        assert stage.ser is not None
        assert stage._running is True

    @patch('serial.Serial')
    def test_init_stage_failure(self, mock_serial):
        """Test stage initialization failure."""
        config = {"id": "COM4", "alias": "TEST"}
        stage = MarlinStageController(config)
        
        # Simulate connection failure
        mock_serial.side_effect = Exception("Port not found")
        
        result = stage.init_stage()
        
        assert result is False
        assert stage.ser is None

    @patch('serial.Serial')
    @patch('threading.Thread')
    def test_deinit_stage(self, mock_thread, mock_serial):
        """Test stage deinitialization."""
        config = {"id": "COM4", "alias": "TEST"}
        stage = MarlinStageController(config)
        
        # Setup mock
        mock_ser = MagicMock()
        mock_serial.return_value = mock_ser
        stage.init_stage()
        
        # Deinitialize
        stage.deinit()
        
        assert stage._running is False
        mock_ser.close.assert_called_once()
        assert stage.ser is None


class TestMarlinStageControllerMovement:
    """Test stage movement operations."""

    def setup_method(self):
        """Setup for each test method."""
        self.config = {
            "id": "COM4",
            "alias": "TEST",
            "settings": {
                "limits_mm": {
                    "x": [0.0, 90.0],
                    "y": [0.0, 100.0]
                }
            }
        }

    @patch('serial.Serial')
    @patch('threading.Thread')
    def test_home_stage(self, mock_thread, mock_serial):
        """Test homing operation."""
        stage = MarlinStageController(self.config)
        
        # Mock serial
        mock_ser = MagicMock()
        mock_serial.return_value = mock_ser
        stage.init_stage()
        
        # Mock position response
        stage._request_position = Mock(return_value=(0.0, 0.0))
        
        with patch('time.sleep'):  # Speed up test
            x, y = stage.home_stage()
        
        assert x == 0.0
        assert y == 0.0
        assert stage._x == 0.0
        assert stage._y == 0.0

    @patch('serial.Serial')
    @patch('threading.Thread')
    def test_move_stage_within_limits(self, mock_thread, mock_serial):
        """Test moving stage to a valid position."""
        stage = MarlinStageController(self.config)
        
        # Mock serial
        mock_ser = MagicMock()
        mock_serial.return_value = mock_ser
        stage.init_stage()
        
        # Mock position response
        stage._request_position = Mock(return_value=(50.0, 50.0))
        
        with patch('time.sleep'):  # Speed up test
            x, y = stage.move_stage(50.0, 50.0)
        
        assert x == 50.0
        assert y == 50.0
        assert stage._x == 50.0
        assert stage._y == 50.0

    @patch('serial.Serial')
    @patch('threading.Thread')
    def test_move_stage_exceeds_x_limit(self, mock_thread, mock_serial):
        """Test that moving beyond X limit raises exception."""
        stage = MarlinStageController(self.config)
        
        # Mock serial
        mock_ser = MagicMock()
        mock_serial.return_value = mock_ser
        stage.init_stage()
        
        with pytest.raises(StageAxisLimitError) as exc_info:
            stage.move_stage(100.0, 50.0)  # X exceeds 90mm
        
        assert exc_info.value.axis == "X"
        assert exc_info.value.value == 100.0
        assert exc_info.value.max_limit == 90.0

    @patch('serial.Serial')
    @patch('threading.Thread')
    def test_move_stage_exceeds_y_limit(self, mock_thread, mock_serial):
        """Test that moving beyond Y limit raises exception."""
        stage = MarlinStageController(self.config)
        
        # Mock serial
        mock_ser = MagicMock()
        mock_serial.return_value = mock_ser
        stage.init_stage()
        
        with pytest.raises(StageAxisLimitError) as exc_info:
            stage.move_stage(50.0, 150.0)  # Y exceeds 100mm
        
        assert exc_info.value.axis == "Y"
        assert exc_info.value.value == 150.0
        assert exc_info.value.max_limit == 100.0

    @patch('serial.Serial')
    @patch('threading.Thread')
    def test_move_stage_negative_position(self, mock_thread, mock_serial):
        """Test that negative positions are rejected."""
        stage = MarlinStageController(self.config)
        
        # Mock serial
        mock_ser = MagicMock()
        mock_serial.return_value = mock_ser
        stage.init_stage()
        
        with pytest.raises(StageAxisLimitError):
            stage.move_stage(-5.0, 50.0)

    @patch('serial.Serial')
    @patch('threading.Thread')
    def test_sample_positions(self, mock_thread, mock_serial):
        """Test moving to sample in and sample out positions."""
        stage = MarlinStageController(self.config)
        
        # Mock serial
        mock_ser = MagicMock()
        mock_serial.return_value = mock_ser
        stage.init_stage()
        
        # Mock position response
        stage._request_position = Mock(side_effect=[
            (57.875, 63.625),  # Sample in
            (90.0, 0.0)         # Sample out
        ])
        
        with patch('time.sleep'):
            # Sample in position from colleague's GUI
            x, y = stage.move_stage(57.875, 63.625)
            assert x == 57.875
            assert y == 63.625
            
            # Sample out position
            x, y = stage.move_stage(90.0, 0.0)
            assert x == 90.0
            assert y == 0.0


class TestMarlinStageControllerPosition:
    """Test position tracking and reporting."""

    @patch('serial.Serial')
    @patch('threading.Thread')
    def test_parse_position_report(self, mock_thread, mock_serial):
        """Test parsing M114 position report."""
        config = {"id": "COM4", "alias": "TEST"}
        stage = MarlinStageController(config)
        
        # Mock serial
        mock_ser = MagicMock()
        mock_serial.return_value = mock_ser
        stage.init_stage()
        
        # Simulate M114 response
        test_response = "X:45.123 Y:67.890 Z:0.000 E:0.000 Count X:451230 Y:678900 Z:0"
        
        # Mock _send_gcode to put response in queue after clearing
        def mock_send_gcode(cmd, wait_for_ok=True, timeout=5.0):
            if "M114" in cmd:
                stage.out_q.put(test_response)
        
        # Request position
        with patch.object(stage, '_send_gcode', side_effect=mock_send_gcode):
            with patch('time.sleep'):
                x, y = stage._request_position()
        
        assert x == 45.123
        assert y == 67.890

    @patch('serial.Serial')
    @patch('threading.Thread')
    def test_get_xy_position(self, mock_thread, mock_serial):
        """Test get_xy_position method."""
        config = {"id": "COM4", "alias": "TEST"}
        stage = MarlinStageController(config)
        
        # Mock serial
        mock_ser = MagicMock()
        mock_serial.return_value = mock_ser
        stage.init_stage()
        
        # Mock _request_position
        stage._request_position = Mock(return_value=(12.345, 67.890))
        
        x, y = stage.get_xy_position()
        
        assert x == 12.345
        assert y == 67.890


class TestMarlinStageControllerGCode:
    """Test G-code command sending."""

    @patch('serial.Serial')
    @patch('threading.Thread')
    def test_send_gcode_single_line(self, mock_thread, mock_serial):
        """Test sending single line G-code."""
        config = {"id": "COM4", "alias": "TEST"}
        stage = MarlinStageController(config)
        
        # Mock serial
        mock_ser = MagicMock()
        mock_serial.return_value = mock_ser
        stage.init_stage()
        
        # Send command without waiting
        with patch('time.sleep'):
            result = stage._send_gcode("G28", wait_for_ok=False)
        
        assert result is True
        mock_ser.write.assert_called_with(b"G28\n")

    @patch('serial.Serial')
    @patch('threading.Thread')
    def test_send_gcode_multiline(self, mock_thread, mock_serial):
        """Test sending multi-line G-code."""
        config = {"id": "COM4", "alias": "TEST"}
        stage = MarlinStageController(config)
        
        # Mock serial
        mock_ser = MagicMock()
        mock_serial.return_value = mock_ser
        stage.init_stage()
        
        # Clear previous calls from init_stage
        mock_ser.write.reset_mock()
        
        # Send multi-line command
        with patch('time.sleep'):
            stage._send_gcode("G90\nG0 X10 Y20", wait_for_ok=False)
        
        calls = mock_ser.write.call_args_list
        assert len(calls) == 2
        assert calls[0][0][0] == b"G90\n"
        assert calls[1][0][0] == b"G0 X10 Y20\n"

    @patch('serial.Serial')
    @patch('threading.Thread')
    def test_send_gcode_wait_for_ok(self, mock_thread, mock_serial):
        """Test sending G-code and waiting for 'ok' response."""
        config = {"id": "COM4", "alias": "TEST"}
        stage = MarlinStageController(config)
        
        # Mock serial
        mock_ser = MagicMock()
        mock_serial.return_value = mock_ser
        stage.init_stage()
        
        # Put 'ok' in queue
        stage.out_q.put("ok")
        
        with patch('time.sleep'):
            result = stage._send_gcode("G90", wait_for_ok=True, timeout=1.0)
        
        assert result is True

    @patch('serial.Serial')
    @patch('threading.Thread')
    def test_emergency_stop(self, mock_thread, mock_serial):
        """Test emergency stop command."""
        config = {"id": "COM4", "alias": "TEST"}
        stage = MarlinStageController(config)
        
        # Mock serial
        mock_ser = MagicMock()
        mock_serial.return_value = mock_ser
        stage.init_stage()
        
        stage.emergency_stop()
        
        mock_ser.write.assert_called_with(b"M112\n")


class TestMarlinStageControllerConfiguration:
    """Test configuration parsing and limits."""

    def test_parse_limits_dict_format(self):
        """Test parsing limits in dict format."""
        config = {
            "id": "COM4",
            "alias": "TEST",
            "settings": {
                "limits_mm": {
                    "x": {"min": 5.0, "max": 85.0},
                    "y": {"min": 10.0, "max": 95.0}
                }
            }
        }
        
        stage = MarlinStageController(config)
        limits = stage.get_limits()
        
        assert limits["x"] == (5.0, 85.0)
        assert limits["y"] == (10.0, 95.0)

    def test_parse_limits_array_format(self):
        """Test parsing limits in array format."""
        config = {
            "id": "COM4",
            "alias": "TEST",
            "settings": {
                "limits_mm": {
                    "x": [0.0, 90.0],
                    "y": [0.0, 100.0]
                }
            }
        }
        
        stage = MarlinStageController(config)
        limits = stage.get_limits()
        
        assert limits["x"] == (0.0, 90.0)
        assert limits["y"] == (0.0, 100.0)

    def test_parse_home_load_positions(self):
        """Test parsing home and load positions."""
        config = {
            "id": "COM4",
            "alias": "TEST",
            "settings": {
                "home": [5.0, 10.0],
                "load": [85.0, 5.0]
            }
        }
        
        stage = MarlinStageController(config)
        positions = stage.get_home_load_positions()
        
        assert positions["home"] == (5.0, 10.0)
        assert positions["load"] == (85.0, 5.0)

    def test_default_positions_when_not_specified(self):
        """Test default home/load positions."""
        config = {"id": "COM4", "alias": "TEST"}
        stage = MarlinStageController(config)
        
        positions = stage.get_home_load_positions()
        assert positions["home"] == (0.0, 0.0)
        assert positions["load"] == (90.0, 0.0)


class TestMarlinStageControllerEdgeCases:
    """Test edge cases and error handling."""

    @patch('serial.Serial')
    @patch('threading.Thread')
    def test_send_gcode_without_connection(self, mock_thread, mock_serial):
        """Test sending G-code without connection raises error."""
        config = {"id": "COM4", "alias": "TEST"}
        stage = MarlinStageController(config)
        
        with pytest.raises(RuntimeError):
            stage._send_gcode("G28")

    def test_movement_time_calculation(self):
        """Test movement time calculation based on distance."""
        config = {"id": "COM4", "alias": "TEST"}
        stage = MarlinStageController(config, feedrate=3000)  # 3000 mm/min = 50 mm/s
        
        stage._x = 0.0
        stage._y = 0.0
        
        # Mock serial to test calculation
        with patch('xystages.serial.Serial'):
            with patch('xystages.threading.Thread'):
                stage.init_stage()
                stage._request_position = Mock(return_value=(50.0, 0.0))
                
                with patch('time.sleep') as mock_sleep:
                    stage.move_stage(50.0, 0.0)
                    
                    # Should sleep for estimated time (50mm / 50mm/s + buffer)
                    # feedrate 3000mm/min = 50mm/s
                    # 50mm distance / 50mm/s = 1s + 0.5s buffer = 1.5s
                    sleep_time = mock_sleep.call_args[0][0]
                    assert 1.0 <= sleep_time <= 2.0

    @patch('serial.Serial')
    @patch('threading.Thread')
    def test_multiple_operations_sequence(self, mock_thread, mock_serial):
        """Test sequence of operations."""
        config = {"id": "COM4", "alias": "TEST"}
        stage = MarlinStageController(config)
        
        # Mock serial
        mock_ser = MagicMock()
        mock_serial.return_value = mock_ser
        stage.init_stage()
        
        # Mock positions
        stage._request_position = Mock(side_effect=[
            (0.0, 0.0),      # After home
            (20.0, 30.0),    # After first move
            (50.0, 60.0),    # After second move
            (0.0, 0.0)       # After return home
        ])
        
        with patch('time.sleep'):
            # Home
            x, y = stage.home_stage()
            assert (x, y) == (0.0, 0.0)
            
            # Move 1
            x, y = stage.move_stage(20.0, 30.0)
            assert (x, y) == (20.0, 30.0)
            
            # Move 2
            x, y = stage.move_stage(50.0, 60.0)
            assert (x, y) == (50.0, 60.0)
            
            # Home again
            x, y = stage.home_stage()
            assert (x, y) == (0.0, 0.0)


if __name__ == "__main__":
    # Allow running tests directly
    pytest.main([__file__, "-v"])
