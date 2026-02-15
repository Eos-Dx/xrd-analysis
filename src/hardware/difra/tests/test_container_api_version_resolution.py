"""Tests for GUI container API version resolution."""

import json
import sys
from pathlib import Path

# Add project src to path
SRC_ROOT = Path(__file__).resolve().parents[3]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from hardware.difra.gui import container_api


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload))


def test_get_container_version_prefers_runtime_config(monkeypatch, tmp_path):
    main_cfg = tmp_path / "main.json"
    global_cfg = tmp_path / "global.json"
    _write_json(main_cfg, {"container_version": "0.2"})
    _write_json(global_cfg, {"container_version": "0.2"})

    monkeypatch.setattr(container_api, "_MAIN_CONFIG_PATH", main_cfg)
    monkeypatch.setattr(container_api, "_GLOBAL_CONFIG_PATH", global_cfg)
    container_api._read_json.cache_clear()

    version = container_api.get_container_version({"container_version": "0.2"})
    assert version == "0.2"


def test_get_container_version_reads_main_json_when_runtime_missing(monkeypatch, tmp_path):
    main_cfg = tmp_path / "main.json"
    global_cfg = tmp_path / "global.json"
    _write_json(main_cfg, {"container_version": "0.2"})
    _write_json(global_cfg, {"container_version": "0.9"})

    monkeypatch.setattr(container_api, "_MAIN_CONFIG_PATH", main_cfg)
    monkeypatch.setattr(container_api, "_GLOBAL_CONFIG_PATH", global_cfg)
    container_api._read_json.cache_clear()

    version = container_api.get_container_version({})
    assert version == "0.2"


def test_get_container_version_uses_global_then_default(monkeypatch, tmp_path):
    main_cfg = tmp_path / "main.json"
    global_cfg = tmp_path / "global.json"
    _write_json(main_cfg, {})
    _write_json(global_cfg, {"container_version": "0.2"})

    monkeypatch.setattr(container_api, "_MAIN_CONFIG_PATH", main_cfg)
    monkeypatch.setattr(container_api, "_GLOBAL_CONFIG_PATH", global_cfg)
    container_api._read_json.cache_clear()

    assert container_api.get_container_version({}) == "0.2"

    _write_json(global_cfg, {})
    container_api._read_json.cache_clear()
    assert container_api.get_container_version({}) == container_api.DEFAULT_CONTAINER_VERSION
