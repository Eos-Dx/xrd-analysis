# OmniScan Certificate Center

Python CLI tool for generating PKI certificates following the OmniSoft Certificate Strategy.

## Features

- **Root CA Generation**: Separate root CAs for Device Servers and Maintenance Clients
- **Server Certificates**: Long-lived (1 year) certificates for OMNIScan hardware devices
- **Client Certificates**: Short-lived certificates for maintenance engineers with device scope
- **Compliance**: ECDSA P-256 or RSA-3072, SHA-256 signatures, aligned with FDA/IEC requirements

## Installation

```bash
pip install cryptography
```

## Directory Structure

```
certs/
├── root/          # Root CA certificates (keep offline/secure)
├── server/        # Server certificates for OMNIScan devices
└── client/        # Client certificates for maintenance engineers
```

## Usage

### 1. Initialize Directory Structure

```bash
python certgen.py init
```

### 2. Create Root CAs

Create the Device Server Root CA:
```bash
python certgen.py create-root --type server
```

Create the Maintenance Client Root CA:
```bash
python certgen.py create-root --type client
```

### 3. Create Server Certificate

```bash
python certgen.py create-server --device-uuid ABC123
```

This creates:
- Server certificate valid for 1 year
- Extended Key Usage: serverAuth
- SAN: `DNS:serverABC123.local`, `URI:urn:omniscan:server:ABC123`

### 4. Create Client Certificate

```bash
python certgen.py create-client --engineer-id ENG001 --device-uuid ABC123 --validity-days 1
```

This creates:
- Client certificate valid for 1 day (default)
- Extended Key Usage: clientAuth
- SAN with device scope: `URI:urn:omniscan:server:ABC123`

## Advanced Options

### Use RSA instead of ECDSA

```bash
python certgen.py create-root --type server --key-type rsa
python certgen.py create-server --device-uuid ABC123 --key-type rsa
```

### Custom Certificate Directory

```bash
python certgen.py init --dir /path/to/custom/dir
python certgen.py create-root --type server --dir /path/to/custom/dir
```

### Short-lived Client Certificate (4 hours)

```bash
python certgen.py create-client --engineer-id ENG001 --device-uuid ABC123 --validity-days 0.166
```

## Certificate Validation

View certificate details:
```bash
openssl x509 -in certs/server/server_ABC123.crt -text -noout
```

Verify certificate chain:
```bash
openssl verify -CAfile certs/root/device_server_root_ca.crt certs/server/server_ABC123.crt
```

## Security Notes

- Root CA private keys should be kept offline and secured
- Server certificates designed for TPM/HSM storage
- Client certificates designed for smartcard/hardware token storage
- Private keys are never encrypted by default (apply HSM/TPM protection in production)

## Compliance

This tool implements the OmniSoft Certificate Strategy aligned with:
- FDA Cybersecurity Guidance (2023)
- IEC 62304 Class B
- ISO 14971, IEC 62366, ISO 27001
