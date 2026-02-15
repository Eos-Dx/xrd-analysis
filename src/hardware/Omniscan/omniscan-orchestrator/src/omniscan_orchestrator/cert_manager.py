"""
Certificate Manager for Engineer Certificates
Wraps the omniscan-certificate-center certgen.py for programmatic certificate generation
"""

import subprocess
import sys
from pathlib import Path
from typing import Optional, Tuple
from datetime import datetime
from cryptography import x509
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import serialization


class CertificateManager:
    """Manages engineer certificates for mTLS authentication"""
    
    def __init__(self, cert_center_dir: Path):
        """
        Initialize the certificate manager
        
        Args:
            cert_center_dir: Path to omniscan-certificate-center directory
        """
        self.cert_center_dir = Path(cert_center_dir)
        self.certgen_script = self.cert_center_dir / "certgen.py"
        self.certs_dir = self.cert_center_dir / "certs"
        
        if not self.certgen_script.exists():
            raise FileNotFoundError(
                f"Certificate generation script not found: {self.certgen_script}"
            )
    
    def ensure_root_ca_exists(self, ca_type: str = "client") -> bool:
        """
        Check if root CA exists, create if it doesn't
        
        Args:
            ca_type: Type of CA ('server' or 'client')
        
        Returns:
            True if CA exists or was created successfully
        """
        ca_name = (
            "device_server_root_ca" if ca_type == "server" 
            else "maintenance_client_root_ca"
        )
        ca_cert_path = self.certs_dir / "root" / f"{ca_name}.crt"
        
        if ca_cert_path.exists():
            return True
        
        print(f"Root CA not found, creating {ca_type} Root CA...")
        result = subprocess.run(
            [
                sys.executable,
                str(self.certgen_script),
                "create-root",
                "--type", ca_type,
                "--dir", str(self.certs_dir),
            ],
            capture_output=True,
            text=True,
        )
        
        if result.returncode != 0:
            print(f"Error creating root CA: {result.stderr}")
            return False
        
        print(result.stdout)
        return True
    
    def create_engineer_certificate(
        self,
        engineer_id: str,
        device_uuid: str,
        validity_days: int = 1,
        key_type: str = "ecdsa",
    ) -> Tuple[Path, Path]:
        """
        Create a new engineer certificate for device access
        
        Args:
            engineer_id: Engineer identifier (e.g., "ENG001")
            device_uuid: Target device UUID
            validity_days: Certificate validity in days (default: 1)
            key_type: Key type - 'ecdsa' or 'rsa' (default: 'ecdsa')
        
        Returns:
            Tuple of (certificate_path, key_path)
        
        Raises:
            RuntimeError: If certificate generation fails
        """
        # Ensure client root CA exists
        if not self.ensure_root_ca_exists("client"):
            raise RuntimeError("Failed to ensure client root CA exists")
        
        print(f"\nGenerating engineer certificate for {engineer_id}...")
        print(f"  Device: {device_uuid}")
        print(f"  Validity: {validity_days} day(s)")
        print(f"  Key Type: {key_type}")
        
        result = subprocess.run(
            [
                sys.executable,
                str(self.certgen_script),
                "create-client",
                "--engineer-id", engineer_id,
                "--device-uuid", device_uuid,
                "--validity-days", str(validity_days),
                "--key-type", key_type,
                "--dir", str(self.certs_dir),
            ],
            capture_output=True,
            text=True,
        )
        
        if result.returncode != 0:
            raise RuntimeError(f"Certificate generation failed: {result.stderr}")
        
        print(result.stdout)
        
        cert_path = (
            self.certs_dir / "client" / f"client_{engineer_id}_{device_uuid}.crt"
        )
        key_path = (
            self.certs_dir / "client" / f"client_{engineer_id}_{device_uuid}.key"
        )
        
        if not cert_path.exists() or not key_path.exists():
            raise RuntimeError("Certificate files not found after generation")
        
        return cert_path, key_path
    
    def get_certificate_info(self, cert_path: Path) -> dict:
        """
        Extract information from a certificate
        
        Args:
            cert_path: Path to certificate file
        
        Returns:
            Dictionary with certificate information
        """
        with open(cert_path, "rb") as f:
            cert = x509.load_pem_x509_certificate(f.read(), default_backend())
        
        subject = cert.subject
        cn = subject.get_attributes_for_oid(x509.oid.NameOID.COMMON_NAME)[0].value
        
        # Extract SAN URIs
        san_uris = []
        try:
            san_ext = cert.extensions.get_extension_for_oid(
                x509.oid.ExtensionOID.SUBJECT_ALTERNATIVE_NAME
            )
            for uri in san_ext.value:
                if isinstance(uri, x509.UniformResourceIdentifier):
                    san_uris.append(uri.value)
        except x509.ExtensionNotFound:
            pass
        
        return {
            "common_name": cn,
            "subject": str(subject),
            "issuer": str(cert.issuer),
            "not_valid_before": cert.not_valid_before_utc,
            "not_valid_after": cert.not_valid_after_utc,
            "serial_number": cert.serial_number,
            "san_uris": san_uris,
            "is_valid": (
                cert.not_valid_before_utc <= datetime.now(cert.not_valid_before_utc.tzinfo)
                <= cert.not_valid_after_utc
            ),
        }
    
    def get_ca_certificate_path(self, ca_type: str = "server") -> Path:
        """
        Get the path to a root CA certificate
        
        Args:
            ca_type: Type of CA ('server' or 'client')
        
        Returns:
            Path to CA certificate
        """
        ca_name = (
            "device_server_root_ca" if ca_type == "server" 
            else "maintenance_client_root_ca"
        )
        return self.certs_dir / "root" / f"{ca_name}.crt"
    
    def list_engineer_certificates(self) -> list[dict]:
        """
        List all engineer certificates in the client directory
        
        Returns:
            List of certificate information dictionaries
        """
        client_dir = self.certs_dir / "client"
        if not client_dir.exists():
            return []
        
        certs = []
        for cert_file in client_dir.glob("client_*.crt"):
            try:
                info = self.get_certificate_info(cert_file)
                info["file_path"] = str(cert_file)
                certs.append(info)
            except Exception as e:
                print(f"Warning: Could not read certificate {cert_file}: {e}")
        
        return certs
