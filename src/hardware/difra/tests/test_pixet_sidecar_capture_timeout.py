import os
import sys

SRC_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if SRC_ROOT not in sys.path:
    sys.path.insert(0, SRC_ROOT)

from hardware.difra.hardware import detectors
from hardware.difra.hardware.detectors import PixetSidecarDetectorController


class _FakeSocket:
    def __init__(self):
        self.timeout_values = []
        self.sent_payload = b""
        self._response = b'{"ok": true, "result": {"captured": true}}\n'

    def settimeout(self, timeout):
        self.timeout_values.append(float(timeout))

    def sendall(self, payload):
        self.sent_payload = payload

    def recv(self, _size):
        response = self._response
        self._response = b""
        return response

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


def test_capture_point_uses_extended_timeout_for_long_exposure(monkeypatch, tmp_path):
    observed_connect_timeouts = []
    fake_socket = _FakeSocket()

    def _fake_create_connection(_address, timeout):
        observed_connect_timeouts.append(float(timeout))
        return fake_socket

    monkeypatch.setattr(detectors.socket, "create_connection", _fake_create_connection)

    controller = PixetSidecarDetectorController(
        alias="PRIMARY",
        config={"pixet_sidecar": {"timeout_s": 10.0, "capture_timeout_pad_s": 30.0}},
    )

    ok = controller.capture_point(
        Nframes=1,
        Nseconds=60.0,
        filename_base=str(tmp_path / "capture"),
    )

    assert ok is True
    assert observed_connect_timeouts
    # 60s exposure + 30s buffer should raise timeout well above the base 10s socket timeout.
    assert observed_connect_timeouts[0] >= 90.0
    assert fake_socket.timeout_values
    assert fake_socket.timeout_values[0] >= 90.0
