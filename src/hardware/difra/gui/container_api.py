"""Resolve container implementation from GUI config (main/global json)."""

import json
from functools import lru_cache
from pathlib import Path
from types import ModuleType
from typing import Any, Dict, Optional

from hardware.container.registry import load_version_module, normalize_version


DEFAULT_CONTAINER_VERSION = "0.2"
_CONFIG_DIR = Path(__file__).resolve().parent.parent / "resources" / "config"
_MAIN_CONFIG_PATH = _CONFIG_DIR / "main.json"
_GLOBAL_CONFIG_PATH = _CONFIG_DIR / "global.json"


@lru_cache(maxsize=4)
def _read_json(path: Path) -> Dict[str, Any]:
    try:
        if not path.exists():
            return {}
        return json.loads(path.read_text())
    except Exception:
        return {}


def _normalize_version_candidate(version: Any) -> Optional[str]:
    if version is None:
        return None
    normalized = str(version).strip()
    return normalized or None


def _read_version_from_file(path: Path) -> Optional[str]:
    config = _read_json(path)
    return _normalize_version_candidate(config.get("container_version"))


def get_container_version(config: Optional[Dict[str, Any]]) -> str:
    # 1) Active merged runtime config (global/setup or legacy config)
    if config:
        version = _normalize_version_candidate(config.get("container_version"))
        if version:
            return version

    # 2) Legacy source-of-truth requested by GUI workflow
    version = _read_version_from_file(_MAIN_CONFIG_PATH)
    if version:
        return version

    # 3) Split global config fallback
    version = _read_version_from_file(_GLOBAL_CONFIG_PATH)
    if version:
        return version

    return DEFAULT_CONTAINER_VERSION


def get_container_module(config: Optional[Dict[str, Any]]) -> ModuleType:
    version = get_container_version(config)
    return load_version_module(normalize_version(version))


def get_schema(config: Optional[Dict[str, Any]]):
    return get_container_module(config).schema


def get_writer(config: Optional[Dict[str, Any]]):
    return get_container_module(config).writer


def get_container_manager(config: Optional[Dict[str, Any]]):
    return get_container_module(config).container_manager


def get_technical_container(config: Optional[Dict[str, Any]]):
    return get_container_module(config).technical_container


def get_technical_validator(config: Optional[Dict[str, Any]]):
    return get_container_module(config).technical_validator


def get_session_container(config: Optional[Dict[str, Any]]):
    return get_container_module(config).session_container
