from __future__ import annotations

import json
from typing import Any, Dict, Optional, Tuple

import requests


class OrchestratorClient:
    def __init__(
        self,
        base_url: str,
        client_cert: Optional[Tuple[str, str]] = None,
        verify: Optional[str | bool] = True,
        timeout: float = 15.0,
        session: Optional[requests.Session] = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.session = session or requests.Session()
        self.client_cert = client_cert
        self.verify = verify
        self.timeout = timeout

    def _req(self, method: str, path: str, **kwargs) -> requests.Response:
        url = f"{self.base_url}/{path.lstrip('/')}"
        kwargs.setdefault("timeout", self.timeout)
        if self.client_cert:
            kwargs.setdefault("cert", self.client_cert)
        if self.verify is not None:
            kwargs.setdefault("verify", self.verify)
        resp = self.session.request(method=method.upper(), url=url, **kwargs)
        resp.raise_for_status()
        return resp

    # Maintenance endpoints
    def status(self) -> Dict[str, Any]:
        return self._req("GET", "/maintenance/status").json()

    def enter_maintenance(self, ttl_seconds: int) -> Dict[str, Any]:
        return self._req("POST", "/maintenance/enter", json={"ttl_seconds": ttl_seconds}).json()

    def renew_maintenance(self, ttl_seconds: int) -> Dict[str, Any]:
        return self._req("POST", "/maintenance/renew", json={"ttl_seconds": ttl_seconds}).json()

    def exit_maintenance(self) -> Dict[str, Any]:
        return self._req("POST", "/maintenance/exit").json()

    # Config endpoints
    def get_config(self) -> Dict[str, Any]:
        return self._req("GET", "/config").json()

    def set_config(self, cfg: Dict[str, Any]) -> Dict[str, Any]:
        return self._req("PUT", "/config", json=cfg).json()

    def patch_config(self, patch: Dict[str, Any]) -> Dict[str, Any]:
        return self._req("PATCH", "/config", json=patch).json()

    def validate_config(self, cfg: Dict[str, Any]) -> Dict[str, Any]:
        return self._req("POST", "/config/validate", json=cfg).json()
