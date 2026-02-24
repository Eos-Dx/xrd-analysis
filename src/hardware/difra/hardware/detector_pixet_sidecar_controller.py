"""PIXet sidecar detector controller."""

import json
import os
import socket
import threading
import time
import uuid
from pathlib import Path
from typing import Optional

import numpy as np

from hardware.difra.hardware.detector_controller_base import DetectorController
from hardware.difra.utils.logger import get_module_logger

logger = get_module_logger(__name__)


class PixetSidecarError(RuntimeError):
    """Raised when sidecar communication or command execution fails."""


class PixetSidecarDetectorController(DetectorController):
    """PIXet controller proxying hardware calls to external socket sidecar."""

    def __init__(self, alias, size=(256, 256), config=None):
        self.alias = alias
        self.size = tuple(size)
        self.config = config or {}
        sidecar_cfg = self.config.get("pixet_sidecar", {}) or {}
        self.sidecar_host = str(
            sidecar_cfg.get(
                "host",
                self.config.get(
                    "sidecar_host",
                    os.environ.get("PIXET_SIDECAR_HOST", "127.0.0.1"),
                ),
            )
        )
        self.sidecar_port = int(
            sidecar_cfg.get(
                "port",
                self.config.get(
                    "sidecar_port",
                    os.environ.get("PIXET_SIDECAR_PORT", "51001"),
                ),
            )
        )
        self.timeout_s = float(
            sidecar_cfg.get(
                "timeout_s",
                self.config.get(
                    "sidecar_timeout_s",
                    os.environ.get("PIXET_SIDECAR_TIMEOUT_S", "10.0"),
                ),
            )
        )
        self.capture_timeout_pad_s = float(
            sidecar_cfg.get(
                "capture_timeout_pad_s",
                self.config.get(
                    "sidecar_capture_timeout_pad_s",
                    os.environ.get("PIXET_SIDECAR_CAPTURE_TIMEOUT_PAD_S", "30.0"),
                ),
            )
        )
        self._stream_thread = None
        self._streaming = threading.Event()

    def _rpc(self, cmd: str, args: dict, timeout_s: Optional[float] = None):
        req_id = str(uuid.uuid4())
        payload = (
            json.dumps(
                {
                    "id": req_id,
                    "cmd": cmd,
                    "args": args,
                },
                ensure_ascii=True,
            )
            + "\n"
        ).encode("utf-8")
        rpc_timeout_s = (
            float(timeout_s) if timeout_s is not None else float(self.timeout_s)
        )
        rpc_timeout_s = max(rpc_timeout_s, 0.1)
        try:
            with socket.create_connection(
                (self.sidecar_host, self.sidecar_port),
                timeout=rpc_timeout_s,
            ) as sock:
                sock.settimeout(rpc_timeout_s)
                sock.sendall(payload)

                response_bytes = b""
                while b"\n" not in response_bytes:
                    chunk = sock.recv(65536)
                    if not chunk:
                        raise PixetSidecarError(
                            "Sidecar closed connection without response"
                        )
                    response_bytes += chunk
        except Exception as e:
            raise PixetSidecarError(
                f"Sidecar connection failed ({self.sidecar_host}:{self.sidecar_port}): {e}"
            )

        line = response_bytes.split(b"\n", 1)[0]
        try:
            response = json.loads(line.decode("utf-8"))
        except Exception as e:
            raise PixetSidecarError(f"Invalid sidecar response JSON: {e}")

        if not response.get("ok"):
            raise PixetSidecarError(response.get("error", "Unknown sidecar error"))
        return response.get("result")

    def init_detector(self):
        logger.info(
            "Initializing Pixet detector via sidecar",
            detector=self.alias,
            sidecar_host=self.sidecar_host,
            sidecar_port=self.sidecar_port,
        )
        result = self._rpc(
            "init_detector",
            {
                "alias": self.alias,
                "size": [int(self.size[0]), int(self.size[1])],
                "config": dict(self.config),
            },
        )
        initialized = bool((result or {}).get("initialized", False))
        detected_size = (result or {}).get("size")
        if (
            isinstance(detected_size, list)
            and len(detected_size) == 2
            and all(isinstance(v, int) for v in detected_size)
        ):
            self.size = (int(detected_size[0]), int(detected_size[1]))
        return initialized

    def capture_point(self, Nframes, Nseconds, filename_base):
        nframes = max(int(Nframes), 1)
        nseconds = float(Nseconds)
        expected_capture_s = max(nseconds, 0.0) * nframes
        capture_timeout_s = max(
            self.timeout_s,
            expected_capture_s + self.capture_timeout_pad_s,
        )
        result = self._rpc(
            "capture_point",
            {
                "alias": self.alias,
                "Nframes": nframes,
                "Nseconds": nseconds,
                "filename_base": str(filename_base),
            },
            timeout_s=capture_timeout_s,
        )
        return bool((result or {}).get("captured", False))

    def deinit_detector(self):
        try:
            self._rpc("deinit_detector", {"alias": self.alias})
        except PixetSidecarError as e:
            logger.warning("Sidecar deinit failed", detector=self.alias, error=str(e))

    def start_stream(self, callback, exposure=0.1, interval=0.0, frames=1):
        self.stop_stream()
        self._streaming.set()
        self._stream_thread = threading.Thread(
            target=self._stream_loop,
            args=(callback, exposure, interval, frames),
            daemon=True,
        )
        self._stream_thread.start()
        logger.info("Sidecar streaming started", detector=self.alias, exposure=exposure)

    def stop_stream(self):
        if self._stream_thread and self._stream_thread.is_alive():
            self._streaming.clear()
            self._stream_thread.join(timeout=2.0)
            logger.info("Sidecar streaming stopped", detector=self.alias)
        self._stream_thread = None

    def _stream_loop(self, callback, exposure, interval, frames):
        while self._streaming.is_set():
            try:
                result = self._rpc(
                    "capture_frame",
                    {
                        "alias": self.alias,
                        "exposure_s": float(exposure),
                        "frames": max(int(frames), 1),
                    },
                )
                frame_payload = (result or {}).get("frame")
                if frame_payload is None:
                    frame = None
                else:
                    frame = np.asarray(frame_payload, dtype=np.float64)
                    if frame.ndim == 2:
                        frame = frame[: self.size[1], : self.size[0]]
                    else:
                        frame = None
                callback({self.alias: frame})
            except Exception as e:
                logger.warning(
                    "Sidecar frame capture error during streaming",
                    detector=self.alias,
                    error=str(e),
                )
                callback({self.alias: None})
            if interval:
                time.sleep(interval)

    def convert_to_container_format(
        self,
        raw_file_path: str,
        container_version: str = "0.2",
    ) -> str:
        raw_path = Path(raw_file_path)
        if container_version == "0.2":
            npy_path = raw_path.with_suffix(".npy")
            if not npy_path.exists():
                try:
                    data = np.loadtxt(raw_path)
                    np.save(npy_path, data)
                except Exception as e:
                    raise RuntimeError(f"Failed to convert {raw_path} to .npy: {e}")
            return str(npy_path)
        raise ValueError(
            f"Detector {self.alias} does not support container version {container_version}"
        )

    def get_raw_file_patterns(self):
        return ["*.txt", "*.dsc"]
