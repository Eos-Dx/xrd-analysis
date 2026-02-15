#!/usr/bin/env python3
"""
OmniScan Certificate Generation CLI
Generates PKI certificates following OmniSoft Certificate Strategy
"""

import argparse
import os
import sys
from pathlib import Path
from datetime import datetime, timedelta
from cryptography import x509
from cryptography.x509.oid import NameOID, ExtendedKeyUsageOID
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa, ec
from cryptography.hazmat.backends import default_backend


class CertificateGenerator:
    def __init__(self, base_dir="certs"):
        self.base_dir = Path(base_dir)
        self.root_dir = self.base_dir / "root"
        self.server_dir = self.base_dir / "server"
        self.client_dir = self.base_dir / "client"
        
    def create_directories(self):
        """Create directory structure for certificates"""
        self.root_dir.mkdir(parents=True, exist_ok=True)
        self.server_dir.mkdir(parents=True, exist_ok=True)
        self.client_dir.mkdir(parents=True, exist_ok=True)
        print(f"✓ Created directory structure in {self.base_dir.absolute()}")
    
    def generate_private_key(self, key_type="ecdsa"):
        """Generate private key (ECDSA P-256 or RSA-3072)"""
        if key_type == "ecdsa":
            return ec.generate_private_key(ec.SECP256R1(), default_backend())
        elif key_type == "rsa":
            return rsa.generate_private_key(
                public_exponent=65537,
                key_size=3072,
                backend=default_backend()
            )
        else:
            raise ValueError("key_type must be 'ecdsa' or 'rsa'")
    
    def save_private_key(self, key, path, password=None):
        """Save private key to file"""
        encryption = serialization.NoEncryption()
        if password:
            encryption = serialization.BestAvailableEncryption(password.encode())
        
        with open(path, "wb") as f:
            f.write(key.private_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PrivateFormat.PKCS8,
                encryption_algorithm=encryption
            ))
    
    def save_certificate(self, cert, path):
        """Save certificate to file"""
        with open(path, "wb") as f:
            f.write(cert.public_bytes(serialization.Encoding.PEM))
    
    def create_root_ca(self, ca_type="server", key_type="ecdsa"):
        """Create Root CA certificate"""
        if ca_type == "server":
            subject_name = "OmniScan Device Server Root CA"
            filename = "device_server_root_ca"
        elif ca_type == "client":
            subject_name = "OmniScan Maintenance Client Root CA"
            filename = "maintenance_client_root_ca"
        else:
            raise ValueError("ca_type must be 'server' or 'client'")
        
        print(f"Generating {subject_name}...")
        
        # Generate private key
        private_key = self.generate_private_key(key_type)
        
        # Create certificate
        subject = issuer = x509.Name([
            x509.NameAttribute(NameOID.COUNTRY_NAME, "US"),
            x509.NameAttribute(NameOID.ORGANIZATION_NAME, "OmniSoft"),
            x509.NameAttribute(NameOID.COMMON_NAME, subject_name),
        ])
        
        cert = (
            x509.CertificateBuilder()
            .subject_name(subject)
            .issuer_name(issuer)
            .public_key(private_key.public_key())
            .serial_number(x509.random_serial_number())
            .not_valid_before(datetime.utcnow())
            .not_valid_after(datetime.utcnow() + timedelta(days=3650))  # 10 years
            .add_extension(
                x509.BasicConstraints(ca=True, path_length=1),
                critical=True,
            )
            .add_extension(
                x509.KeyUsage(
                    digital_signature=True,
                    key_cert_sign=True,
                    crl_sign=True,
                    key_encipherment=False,
                    content_commitment=False,
                    data_encipherment=False,
                    key_agreement=False,
                    encipher_only=False,
                    decipher_only=False,
                ),
                critical=True,
            )
            .sign(private_key, hashes.SHA256(), default_backend())
        )
        
        # Save files
        key_path = self.root_dir / f"{filename}.key"
        cert_path = self.root_dir / f"{filename}.crt"
        
        self.save_private_key(private_key, key_path)
        self.save_certificate(cert, cert_path)
        
        print(f"✓ Root CA created:")
        print(f"  Private Key: {key_path}")
        print(f"  Certificate: {cert_path}")
        
        return private_key, cert
    
    def create_server_certificate(self, device_uuid, root_key=None, root_cert=None, key_type="ecdsa"):
        """Create Server certificate for OMNIScan device"""
        print(f"Generating server certificate for device {device_uuid}...")
        
        # Load root CA if not provided
        if root_key is None or root_cert is None:
            root_key_path = self.root_dir / "device_server_root_ca.key"
            root_cert_path = self.root_dir / "device_server_root_ca.crt"
            
            if not root_key_path.exists() or not root_cert_path.exists():
                print("ERROR: Root CA not found. Please create root CA first.")
                return
            
            with open(root_key_path, "rb") as f:
                root_key = serialization.load_pem_private_key(
                    f.read(), password=None, backend=default_backend()
                )
            
            with open(root_cert_path, "rb") as f:
                root_cert = x509.load_pem_x509_certificate(f.read(), default_backend())
        
        # Generate private key
        private_key = self.generate_private_key(key_type)
        
        # Create certificate
        subject = x509.Name([
            x509.NameAttribute(NameOID.COUNTRY_NAME, "US"),
            x509.NameAttribute(NameOID.ORGANIZATION_NAME, "OmniSoft"),
            x509.NameAttribute(NameOID.COMMON_NAME, f"OMNIScan Server {device_uuid}"),
        ])
        
        cert = (
            x509.CertificateBuilder()
            .subject_name(subject)
            .issuer_name(root_cert.subject)
            .public_key(private_key.public_key())
            .serial_number(x509.random_serial_number())
            .not_valid_before(datetime.utcnow())
            .not_valid_after(datetime.utcnow() + timedelta(days=365))  # 1 year
            .add_extension(
                x509.BasicConstraints(ca=False, path_length=None),
                critical=True,
            )
            .add_extension(
                x509.KeyUsage(
                    digital_signature=True,
                    key_encipherment=True,
                    key_cert_sign=False,
                    crl_sign=False,
                    content_commitment=False,
                    data_encipherment=False,
                    key_agreement=False,
                    encipher_only=False,
                    decipher_only=False,
                ),
                critical=True,
            )
            .add_extension(
                x509.ExtendedKeyUsage([ExtendedKeyUsageOID.SERVER_AUTH]),
                critical=True,
            )
            .add_extension(
                x509.SubjectAlternativeName([
                    x509.DNSName(f"server{device_uuid}.local"),
                    x509.UniformResourceIdentifier(f"urn:omniscan:server:{device_uuid}"),
                ]),
                critical=False,
            )
            .sign(root_key, hashes.SHA256(), default_backend())
        )
        
        # Save files
        key_path = self.server_dir / f"server_{device_uuid}.key"
        cert_path = self.server_dir / f"server_{device_uuid}.crt"
        
        self.save_private_key(private_key, key_path)
        self.save_certificate(cert, cert_path)
        
        print(f"✓ Server certificate created:")
        print(f"  Private Key: {key_path}")
        print(f"  Certificate: {cert_path}")
    
    def create_client_certificate(self, engineer_id, device_uuid, validity_days=1, root_key=None, root_cert=None, key_type="ecdsa"):
        """Create Client certificate for maintenance engineer"""
        print(f"Generating client certificate for engineer {engineer_id} (device: {device_uuid})...")
        
        # Load root CA if not provided
        if root_key is None or root_cert is None:
            root_key_path = self.root_dir / "maintenance_client_root_ca.key"
            root_cert_path = self.root_dir / "maintenance_client_root_ca.crt"
            
            if not root_key_path.exists() or not root_cert_path.exists():
                print("ERROR: Root CA not found. Please create root CA first.")
                return
            
            with open(root_key_path, "rb") as f:
                root_key = serialization.load_pem_private_key(
                    f.read(), password=None, backend=default_backend()
                )
            
            with open(root_cert_path, "rb") as f:
                root_cert = x509.load_pem_x509_certificate(f.read(), default_backend())
        
        # Generate private key
        private_key = self.generate_private_key(key_type)
        
        # Create certificate
        subject = x509.Name([
            x509.NameAttribute(NameOID.COUNTRY_NAME, "US"),
            x509.NameAttribute(NameOID.ORGANIZATION_NAME, "OmniSoft"),
            x509.NameAttribute(NameOID.COMMON_NAME, f"Maintenance Engineer {engineer_id}"),
        ])
        
        cert = (
            x509.CertificateBuilder()
            .subject_name(subject)
            .issuer_name(root_cert.subject)
            .public_key(private_key.public_key())
            .serial_number(x509.random_serial_number())
            .not_valid_before(datetime.utcnow())
            .not_valid_after(datetime.utcnow() + timedelta(days=validity_days))
            .add_extension(
                x509.BasicConstraints(ca=False, path_length=None),
                critical=True,
            )
            .add_extension(
                x509.KeyUsage(
                    digital_signature=True,
                    key_encipherment=True,
                    key_cert_sign=False,
                    crl_sign=False,
                    content_commitment=False,
                    data_encipherment=False,
                    key_agreement=False,
                    encipher_only=False,
                    decipher_only=False,
                ),
                critical=True,
            )
            .add_extension(
                x509.ExtendedKeyUsage([ExtendedKeyUsageOID.CLIENT_AUTH]),
                critical=True,
            )
            .add_extension(
                x509.SubjectAlternativeName([
                    x509.UniformResourceIdentifier(f"urn:omniscan:server:{device_uuid}"),
                ]),
                critical=False,
            )
            .sign(root_key, hashes.SHA256(), default_backend())
        )
        
        # Save files
        key_path = self.client_dir / f"client_{engineer_id}_{device_uuid}.key"
        cert_path = self.client_dir / f"client_{engineer_id}_{device_uuid}.crt"
        
        self.save_private_key(private_key, key_path)
        self.save_certificate(cert, cert_path)
        
        print(f"✓ Client certificate created (valid for {validity_days} day(s)):")
        print(f"  Private Key: {key_path}")
        print(f"  Certificate: {cert_path}")


def main():
    parser = argparse.ArgumentParser(
        description="OmniScan Certificate Generation CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")
    
    # Init command
    init_parser = subparsers.add_parser("init", help="Initialize certificate directory structure")
    init_parser.add_argument("--dir", default="certs", help="Base directory for certificates (default: certs)")
    
    # Root CA command
    root_parser = subparsers.add_parser("create-root", help="Create Root CA certificate")
    root_parser.add_argument("--type", choices=["server", "client"], required=True, 
                             help="Type of Root CA (server or client)")
    root_parser.add_argument("--key-type", choices=["ecdsa", "rsa"], default="ecdsa",
                             help="Key type (default: ecdsa)")
    root_parser.add_argument("--dir", default="certs", help="Base directory for certificates (default: certs)")
    
    # Server certificate command
    server_parser = subparsers.add_parser("create-server", help="Create server certificate")
    server_parser.add_argument("--device-uuid", required=True, help="Device UUID")
    server_parser.add_argument("--key-type", choices=["ecdsa", "rsa"], default="ecdsa",
                               help="Key type (default: ecdsa)")
    server_parser.add_argument("--dir", default="certs", help="Base directory for certificates (default: certs)")
    
    # Client certificate command
    client_parser = subparsers.add_parser("create-client", help="Create client certificate")
    client_parser.add_argument("--engineer-id", required=True, help="Engineer ID")
    client_parser.add_argument("--device-uuid", required=True, help="Device UUID for scope")
    client_parser.add_argument("--validity-days", type=int, default=1, 
                               help="Certificate validity in days (default: 1)")
    client_parser.add_argument("--key-type", choices=["ecdsa", "rsa"], default="ecdsa",
                               help="Key type (default: ecdsa)")
    client_parser.add_argument("--dir", default="certs", help="Base directory for certificates (default: certs)")
    
    args = parser.parse_args()
    
    if args.command is None:
        parser.print_help()
        return
    
    gen = CertificateGenerator(args.dir)
    
    if args.command == "init":
        gen.create_directories()
    
    elif args.command == "create-root":
        gen.create_directories()
        gen.create_root_ca(ca_type=args.type, key_type=args.key_type)
    
    elif args.command == "create-server":
        gen.create_server_certificate(args.device_uuid, key_type=args.key_type)
    
    elif args.command == "create-client":
        gen.create_client_certificate(
            args.engineer_id, 
            args.device_uuid, 
            args.validity_days,
            key_type=args.key_type
        )


if __name__ == "__main__":
    main()
