#!/usr/bin/env python3
"""PIXet sidecar server (Python 3.7+) using plain TCP sockets + JSON lines.

Protocol (request/response):
- Request:  {"id": "...", "cmd": "...", "args": {...}}
- Response: {"id": "...", "ok": true,  "result": ...}
            {"id": "...", "ok": false, "error": "..."}
"""

from __future__ import annotations

import argparse
import json
import os
import socketserver
import tempfile
import threading
from pathlib import Path
from typing import Any, Dict

import numpy as np

# Allow running directly from repository root or arbitrary working directory.
REPO_ROOT = Path(__file__).resolve().parents[4]
SRC_ROOT = REPO_ROOT / "src"
import sys

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from hardware.difra.hardware.detectors import (
    DummyDetectorController,
    PixetLegacyDetectorController,
)


class SidecarState:
    def __init__(self) -> None:
        self.controllers: Dict[str, Any] = {}
        # pypixet runtime is effectively process-global; serialize all hardware ops.
        self.op_lock = threading.RLock()


STATE = SidecarState()


def _require_alias(args: Dict[str, Any]) -> str:
    alias = str(args.get("alias", "")).strip()
    if not alias:
        raise ValueError("Missing required field: alias")
    return alias


def _resolve_detector_kind(args: Dict[str, Any]) -> str:
    detector_type = str(args.get("detector_type", "")).strip()
    if not detector_type:
        cfg = args.get("config", {}) or {}
        if isinstance(cfg, dict):
            detector_type = str(cfg.get("type", "")).strip()
    detector_type = detector_type.lower()
    if detector_type in {"dummydetector", "dummy"}:
        return "dummy"
    # Default to legacy pixet path for physical detectors.
    return "pixet_legacy"


def _get_or_create_controller(args: Dict[str, Any]):
    alias = _require_alias(args)
    with STATE.op_lock:
        ctrl = STATE.controllers.get(alias)
        if ctrl is not None:
            return ctrl

        size = args.get("size", [256, 256])
        if not isinstance(size, (list, tuple)) or len(size) != 2:
            raise ValueError("size must be [width, height]")
        width = int(size[0])
        height = int(size[1])

        config = args.get("config", {}) or {}
        if not isinstance(config, dict):
            raise ValueError("config must be an object")

        kind = _resolve_detector_kind(args)
        if kind == "dummy":
            ctrl = DummyDetectorController(alias=alias, size=(width, height))
        else:
            ctrl = PixetLegacyDetectorController(
                alias=alias,
                size=(width, height),
                config=config,
            )
        STATE.controllers[alias] = ctrl
        return ctrl


def _cleanup_temp_capture(base_path: str) -> None:
    for suffix in (".txt", ".dsc"):
        try:
            os.remove(base_path + suffix)
        except OSError:
            pass


def _dispatch(cmd: str, args: Dict[str, Any]) -> Any:
    if cmd == "ping":
        return {"status": "ok", "pid": os.getpid()}

    if cmd == "init_detector":
        ctrl = _get_or_create_controller(args)
        with STATE.op_lock:
            ok = bool(ctrl.init_detector())
            return {
                "initialized": ok,
                "size": [int(ctrl.size[0]), int(ctrl.size[1])],
            }

    if cmd == "deinit_detector":
        alias = _require_alias(args)
        with STATE.op_lock:
            ctrl = STATE.controllers.pop(alias, None)
            if ctrl is not None:
                ctrl.deinit_detector()
        return {"deinitialized": True}

    if cmd == "capture_point":
        ctrl = _get_or_create_controller(args)
        nframes = max(int(args.get("Nframes", 1)), 1)
        nseconds = float(args.get("Nseconds", 0.1))
        filename_base = str(args.get("filename_base", "")).strip()
        if not filename_base:
            raise ValueError("Missing required field: filename_base")

        with STATE.op_lock:
            ok = bool(
                ctrl.capture_point(
                    Nframes=nframes,
                    Nseconds=nseconds,
                    filename_base=filename_base,
                )
            )
        return {"captured": ok}

    if cmd == "capture_frame":
        ctrl = _get_or_create_controller(args)
        exposure_s = float(args.get("exposure_s", 0.1))
        frames = max(int(args.get("frames", 1)), 1)

        # Use normal capture path and return matrix to caller for real-time preview.
        fd, tmp_txt = tempfile.mkstemp(prefix="pixet_sidecar_rt_", suffix=".txt")
        os.close(fd)
        base_path = tmp_txt[:-4]
        _cleanup_temp_capture(base_path)

        try:
            with STATE.op_lock:
                ok = bool(
                    ctrl.capture_point(
                        Nframes=frames,
                        Nseconds=exposure_s,
                        filename_base=base_path,
                    )
                )
            if not ok or not os.path.exists(tmp_txt):
                return {"frame": None}
            frame = np.loadtxt(tmp_txt)
            return {"frame": frame.tolist()}
        finally:
            _cleanup_temp_capture(base_path)

    if cmd == "shutdown":
        with STATE.op_lock:
            aliases = list(STATE.controllers.keys())
            for alias in aliases:
                ctrl = STATE.controllers.pop(alias, None)
                if ctrl is not None:
                    ctrl.deinit_detector()
        return {"shutdown": True}

    raise ValueError("Unknown command: %s" % cmd)


class JsonLineHandler(socketserver.StreamRequestHandler):
    def handle(self) -> None:
        while True:
            raw = self.rfile.readline()
            if not raw:
                return
            req_id = None
            try:
                req = json.loads(raw.decode("utf-8"))
                req_id = req.get("id")
                cmd = str(req.get("cmd", "")).strip()
                args = req.get("args", {}) or {}
                if not isinstance(args, dict):
                    raise ValueError("args must be an object")
                result = _dispatch(cmd, args)
                resp = {"id": req_id, "ok": True, "result": result}
            except Exception as exc:
                resp = {"id": req_id, "ok": False, "error": str(exc)}

            payload = (json.dumps(resp, ensure_ascii=True) + "\n").encode("utf-8")
            self.wfile.write(payload)
            self.wfile.flush()


class ThreadedTCPServer(socketserver.ThreadingMixIn, socketserver.TCPServer):
    allow_reuse_address = True
    daemon_threads = True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run PIXet socket sidecar server.")
    parser.add_argument("--host", default="127.0.0.1", help="Bind host.")
    parser.add_argument("--port", type=int, default=51001, help="Bind port.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    with ThreadedTCPServer((args.host, int(args.port)), JsonLineHandler) as server:
        print(
            "[pixet-sidecar] listening on %s:%s pid=%s"
            % (args.host, args.port, os.getpid())
        )
        try:
            server.serve_forever()
        finally:
            try:
                _dispatch("shutdown", {})
            except Exception:
                pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
