# OmniSoft Certificate Strategy

## Overview
OmniSoft uses a Public Key Infrastructure (PKI) with strict separation of authorities and roles to meet FDA cybersecurity and IEC 62304 Class B requirements. This design ensures that each device and maintenance engineer has a unique, verifiable identity for mutual TLS (mTLS) communication, full auditability, and secure lifecycle management.

## 1. PKI Hierarchy
- **Root CAs (offline):**
  - Device Server Root CA – issues certificates for OMNIScan hardware servers.
  - Maintenance Client Root CA – issues certificates for maintenance clients.
- **Intermediates (online):**
  - Each root issues one intermediate CA for daily signing operations.
  - Roots remain offline for security.

## 2. Certificate Types
| Certificate | Purpose | EKU | Example SAN | Policy OID |
|--------------|----------|-----|--------------|-------------|
| Server cert | Identifies OMNIScan hardware | serverAuth | DNS:server123.local, URI:urn:omniscan:server:UUID | DeviceServer |
| Client cert | Authenticates maintenance engineer | clientAuth | URI:urn:omniscan:server:UUID | MaintenanceAccess |

- Server certs are long-lived (1 year) and stored in TPM/HSM.
- Client certs are short-lived (hours–days) and stored in smartcards or OS keystores.

## 3. Authorization Logic
- mTLS connection: both sides authenticate using their certificates.
- Server verifies:
  1. Certificate chain and policy OID.
  2. Scope (matches its own device UUID).
  3. Certificate validity period and revocation status (OCSP/CRL).
- Optional second factor (PIN/biometric) for maintenance mode entry.

## 4. Maintenance Mode Flow
1. mTLS connection established.
2. Server checks policy and scope.
3. If valid, enters *exclusive maintenance lock* with TTL (renewable).
4. All actions are logged (cert fingerprint, subject, scope, timestamp, result).
5. Auto-exit when TTL expires or session ends.

## 5. Cryptography Standards
- TLS 1.3 (TLS 1.2 fallback allowed).
- ECDSA P‑256 or RSA‑3072 keys, SHA‑256 signatures.
- AEAD cipher suites only (AES‑GCM or ChaCha20‑Poly1305).
- FIPS 140‑3 validated crypto libraries.

## 6. Key Protection
- Server private keys sealed in TPM or HSM.
- Client keys in smartcards or hardware tokens (YubiKey, PIV).
- Exporting private keys is forbidden.

## 7. Rotation and Revocation
- Client certs rotated frequently (hours–days).
- Server certs rotated annually with overlap.
- OCSP primary, CRL fallback.
- Intermediates rotated on defined schedule with staged trust.

## 8. Auditing and Traceability
- All maintenance actions logged with hash chaining for tamper evidence.
- Logs time-synced, signed daily, and exportable to SIEM via TLS.

## 9. Development and Test Mode
- Allow pinned client certs per test server.
- Disable OCSP/CRL checks.
- Short lifetimes and clearly marked non-production OIDs.

## 10. Compliance Mapping
This strategy supports:
- **FDA Cybersecurity Guidance (2023)**
- **ISO 14971** — Risk management
- **IEC 62304** — Secure software lifecycle
- **IEC 62366** — Human factors (for maintenance access)
- **ISO 27001** — Security management

---
**Summary:**  
Every OMNIScan device and maintenance engineer has its own verifiable digital identity. Access is granted only via mTLS using short‑lived, policy‑bound certificates. Each session is cryptographically audited, ensuring safety, accountability, and regulatory compliance.
