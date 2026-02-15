from __future__ import annotations

import json
import os
import pathlib
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from typing import Optional, Tuple

from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.x509.oid import NameOID


STUB_CERT_NAME = "maintenance_cert.pem"
STUB_KEY_NAME = "maintenance_key.pem"
STUB_META_NAME = "metadata.json"


@dataclass(frozen=True)
class USBIdentity:
    cert_path: pathlib.Path
    key_path: pathlib.Path
    ca_bundle_path: Optional[pathlib.Path]
    fingerprint_sha256: str
    metadata: dict

    def requests_cert(self) -> Tuple[str, str]:
        return (str(self.cert_path), str(self.key_path))


class MaintenanceUSBStub:
    """
    File-based stub that emulates a maintenance USB token.

    Layout:
      <root>/maintenance_cert.pem   # PEM-encoded client certificate
      <root>/maintenance_key.pem    # PEM-encoded private key (unencrypted; dev only)
      <root>/metadata.json          # metadata including device UUID and fingerprint
      <root>/ca_bundle.pem          # optional bundle used to verify server (dev)
    """

    def __init__(self, root: os.PathLike | str):
        self.root = pathlib.Path(root)
        if not self.root.exists():
            raise FileNotFoundError(f"USB stub path not found: {self.root}")

        self.cert_path = self.root / STUB_CERT_NAME
        self.key_path = self.root / STUB_KEY_NAME
        self.meta_path = self.root / STUB_META_NAME
        self.ca_bundle_path = self.root / "ca_bundle.pem"

        for p in [self.cert_path, self.key_path, self.meta_path]:
            if not p.exists():
                raise FileNotFoundError(f"Missing required file in USB stub: {p}")

        # compute fingerprint
        with open(self.cert_path, "rb") as f:
            cert = x509.load_pem_x509_certificate(f.read())
        fp = cert.fingerprint(hashes.SHA256()).hex()

        with open(self.meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        # Ensure fingerprint present in metadata for audit
        meta.setdefault("fingerprint_sha256", fp)

        object.__setattr__(self, "identity", USBIdentity(
            cert_path=self.cert_path,
            key_path=self.key_path,
            ca_bundle_path=self.ca_bundle_path if self.ca_bundle_path.exists() else None,
            fingerprint_sha256=fp,
            metadata=meta,
        ))

    @staticmethod
    def create_dev_stub(
        out_dir: os.PathLike | str,
        server_uuid: str,
        common_name: str | None = None,
        days_valid: int = 30,
    ) -> "MaintenanceUSBStub":
        """
        Create a development-only USB stub with a self-signed client certificate.
        For production, certificates must be issued by the Maintenance Client CA and keys stored in hardware.
        """
        out = pathlib.Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)

        key = ec.generate_private_key(ec.SECP256R1())
        subject = issuer = x509.Name([
            x509.NameAttribute(NameOID.ORGANIZATION_NAME, "Omniscan Dev"),
            x509.NameAttribute(NameOID.COMMON_NAME, common_name or f"maint-{server_uuid}"),
        ])
        now = datetime.now(timezone.utc)
        cert = (
            x509.CertificateBuilder()
            .subject_name(subject)
            .issuer_name(issuer)
            .public_key(key.public_key())
            .serial_number(x509.random_serial_number())
            .not_valid_before(now)
            .not_valid_after(now + timedelta(days=days_valid))
            .add_extension(x509.BasicConstraints(ca=False, path_length=None), critical=True)
            .add_extension(x509.SubjectAlternativeName([x509.UniformResourceIdentifier(f"urn:omniscan:server:{server_uuid}")]), critical=False)
            .sign(private_key=key, algorithm=hashes.SHA256())
        )

        cert_path = out / STUB_CERT_NAME
        key_path = out / STUB_KEY_NAME
        meta_path = out / STUB_META_NAME

        with open(cert_path, "wb") as f:
            f.write(cert.public_bytes(serialization.Encoding.PEM))
        with open(key_path, "wb") as f:
            f.write(
                key.private_bytes(
                    encoding=serialization.Encoding.PEM,
                    format=serialization.PrivateFormat.PKCS8,
                    encryption_algorithm=serialization.NoEncryption(),  # dev only
                )
            )
        fp = cert.fingerprint(hashes.SHA256()).hex()
        meta = {
            "type": "dev-maintenance-usb-stub",
            "server_uuid": server_uuid,
            "common_name": common_name or f"maint-{server_uuid}",
            "created_at": now.isoformat(),
            "fingerprint_sha256": fp,
            "policy": "DEV-NON-PROD",
        }
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

        # Set restrictive perms where possible
        try:
            os.chmod(key_path, 0o600)
        except Exception:
            pass

        return MaintenanceUSBStub(out)

    def load(self) -> USBIdentity:
        return self.identity
