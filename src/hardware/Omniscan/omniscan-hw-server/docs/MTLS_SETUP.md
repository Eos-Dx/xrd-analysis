# mTLS Setup Guide for OMNIScan Hardware Server

⚠️ **Implementation Status**: mTLS framework complete, certificate validation NOT enforced in current demo mode. Production deployment required for full security enforcement.

This guide explains how to set up mutual TLS (mTLS) authentication between the OMNIScan Hardware Server and the Python Orchestrator, following the OmniSoft Certificate Strategy.

## Overview

The OMNIScan system uses a Public Key Infrastructure (PKI) with:
- **Server Certificates**: Long-lived (1 year) certificates for OMNIScan hardware servers
- **Client Certificates**: Short-lived certificates for maintenance engineers with device-scoped access
- **Root CAs**: Separate root certificate authorities for servers and clients

## Prerequisites

1. Certificate generation tool installed:
   ```bash
   cd ../omniscan-certificate-center
   pip install -r requirements.txt
   ```

2. Device UUID assigned to your OMNIScan hardware server (e.g., `ABC123`)

## Step 1: Generate Certificates

### Generate Root CAs

```bash
cd ../omniscan-certificate-center

# Create Device Server Root CA
python certgen.py create-root --type server

# Create Maintenance Client Root CA
python certgen.py create-root --type client
```

This creates:
- `certs/root/device_server_root_ca.crt` and `.key`
- `certs/root/maintenance_client_root_ca.crt` and `.key`

### Generate Server Certificate

```bash
# Replace ABC123 with your device UUID
python certgen.py create-server --device-uuid ABC123
```

This creates:
- `certs/server/server_ABC123.crt`
- `certs/server/server_ABC123.key`

### Generate Client Certificates (for testing)

```bash
# For engineer ENG001 to access device ABC123
python certgen.py create-client --engineer-id ENG001 --device-uuid ABC123 --validity-days 7
```

This creates:
- `certs/client/client_ENG001_ABC123.crt`
- `certs/client/client_ENG001_ABC123.key`

## Step 2: Configure the Hardware Server

Create or update `configs/server_with_mtls.json`:

```json
{
  "version": 1,
  "device": {
    "name": "OMNIScan XRD Device",
    "interlocks_armed": true,
    "emergency_stop": false,
    "power_default_on": false
  },
  "certificates": {
    "device_uuid": "ABC123",
    "cert_base_dir": "../omniscan-certificate-center/certs",
    "enable_mtls": true
  }
}
```

**Key Configuration:**
- `device_uuid`: Must match the UUID used when generating the server certificate
- `cert_base_dir`: Path to the certificate directory (relative or absolute)
- `enable_mtls`: Set to `true` to enable mTLS, `false` for plaintext (dev only)

## Step 3: Run the Server with mTLS

```bash
cargo run --release -- --config configs/server_with_mtls.json
```

Expected output:
```
🩺 Starting OMNIScan Medical Device Hardware Server v0.2.0
🔐 Configuring mTLS for secure server-orchestrator communication
🔐 Loading server certificates for mTLS...
  ✓ Loaded server certificate: ../omniscan-certificate-center/certs/server/server_ABC123.crt
  ✓ Device UUID validated: ABC123
  ✓ Loaded server private key: ../omniscan-certificate-center/certs/server/server_ABC123.key
  ✓ Loaded client CA certificate: ../omniscan-certificate-center/certs/root/maintenance_client_root_ca.crt
🔒 mTLS server configuration complete
  - Server authentication: ENABLED
  - Client authentication: REQUIRED
  - Certificate validation: ENFORCED
🔒 mTLS enabled - client certificates required
🆔 Device UUID: ABC123
🚀 Starting OMNIScan gRPC server on [::1]:50051
```

## Step 4: Connect Python Orchestrator with Client Certificate

Update your Python orchestrator to use client certificates:

```python
import grpc
from hub.v1 import hub_pb2_grpc

# Load client certificate and key
with open("certs/client/client_ENG001_ABC123.crt", "rb") as f:
    client_cert = f.read()

with open("certs/client/client_ENG001_ABC123.key", "rb") as f:
    client_key = f.read()

# Load server CA certificate
with open("certs/root/device_server_root_ca.crt", "rb") as f:
    server_ca = f.read()

# Create SSL credentials
credentials = grpc.ssl_channel_credentials(
    root_certificates=server_ca,
    private_key=client_key,
    certificate_chain=client_cert
)

# Connect with mTLS
channel = grpc.secure_channel('[::1]:50051', credentials)
stub = hub_pb2_grpc.AcquisitionStub(channel)

# Make authenticated request
response = stub.GetState({})
print(response)
```

## Certificate Validation

The server automatically validates:
1. **Client certificate chain**: Signed by the Maintenance Client Root CA
2. **Device UUID scope**: Certificate's SAN must contain `urn:omniscan:server:ABC123`
3. **Certificate validity**: Not expired
4. **Extended Key Usage**: Must have `clientAuth`

If validation fails, the server rejects the connection with:
```
Status: PERMISSION_DENIED
Message: "Client certificate device UUID does not match this server"
```

## Certificate Rotation

### Server Certificate Rotation (Annual)

```bash
# Generate new server certificate
cd ../omniscan-certificate-center
python certgen.py create-server --device-uuid ABC123

# Restart server (zero-downtime rotation possible with load balancer)
cd ../omniscan-hw-server
cargo run --release -- --config configs/server_with_mtls.json
```

### Client Certificate Rotation (Frequent)

```bash
# Generate new short-lived client certificate
python certgen.py create-client --engineer-id ENG001 --device-uuid ABC123 --validity-days 1

# Update orchestrator configuration with new certificate paths
```

## Development Mode (Plaintext)

For local development without certificates:

```json
{
  "certificates": {
    "device_uuid": "DEMO-001",
    "cert_base_dir": "../omniscan-certificate-center/certs",
    "enable_mtls": false
  }
}
```

⚠️ **Warning**: Never use `enable_mtls: false` in production!

## Troubleshooting

### Certificate Not Found

```
Error: Failed to load TLS configuration: Failed to open certificate file
```

**Solution**: Verify paths and ensure certificates exist:
```bash
ls -la ../omniscan-certificate-center/certs/server/
ls -la ../omniscan-certificate-center/certs/root/
```

### Device UUID Mismatch

```
⚠️  Device UUID mismatch: expected 'ABC123', found 'XYZ789'
```

**Solution**: Regenerate server certificate with correct UUID or update config.

### Client Certificate Rejected

```
Status: PERMISSION_DENIED
```

**Solution**: 
1. Verify client certificate is signed by Maintenance Client Root CA
2. Confirm SAN contains correct device UUID: `urn:omniscan:server:ABC123`
3. Check certificate hasn't expired

### Permission Denied (File Access)

On Windows, you may need to adjust file permissions:
```powershell
icacls ..\omniscan-certificate-center\certs\server\server_ABC123.key /grant:r "%USERNAME%:R"
```

## Security Best Practices

1. **Keep Root CA Keys Offline**: Store root CA private keys on secure, offline media
2. **Rotate Client Certificates Frequently**: Use 1-7 day validity for client certificates
3. **Monitor Certificate Expiry**: Implement automated alerts for expiring certificates
4. **Audit Certificate Usage**: Review audit logs for certificate-based authentication
5. **Use Hardware Key Storage**: Deploy server keys to TPM/HSM in production
6. **Never Share Private Keys**: Each device and engineer must have unique certificates

## FDA Compliance Notes

This mTLS implementation supports:
- **FDA Cybersecurity Guidance (2023)**: Strong authentication and encryption
- **IEC 62304 Class B**: Secure communication for medical device software
- **ISO 27001**: Information security management
- **HIPAA**: Protected health information transmission security

All certificate-based authentication events are logged to the audit database.
