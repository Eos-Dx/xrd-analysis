# Requirements Traceability Matrix
# Omniscan Hardware Server - Software User Requirements

**Document Version:** 1.0  
**Date:** October 25, 2025  
**Purpose:** Map SOFTWARE_USER_REQUIREMENTS_PROFESSIONAL.md to Doorstop requirement structure

---

## Mapping Summary

This document provides complete traceability from the professional user requirements document to the Doorstop-managed requirements in the REQ/ directory structure.

### Directory Structure
- `usr-omni-server/` - User requirements (USR_OMNI-SERVER_xxx.yml)
- `sys-omni-server/` - System requirements (SYS_OMNI-SERVER_xxx.yml)
- `ver-omni-server/` - Verification requirements (VER_OMNI-SERVER_xxx.yml)
- `risk-omni-server/` - Risk control requirements (RISK_OMNI-SERVER_xxx.yml)

---

## Detailed Traceability Mapping

### 1. Safety and Radiation Protection Requirements

| Professional Req | Doorstop User Req | Doorstop System Req | Risk Control | Verification |
|-----------------|-------------------|---------------------|--------------|--------------|
| **UR-SAFE-001** Operator Safety Protection | USR_OMNI-SERVER_001 | SYS_OMNI-SERVER_001, 002, 003 | RISK_OMNI-SERVER_001, 002, 003 | VER_OMNI-SERVER_001, 003 |
| **UR-SAFE-002** Maintenance Mode Safety | USR_OMNI-SERVER_004 | SYS_OMNI-SERVER_011 | RISK_OMNI-SERVER_006 | VER_OMNI-SERVER_006 |

### 2. X-ray Exposure Control

| Professional Req | Doorstop User Req | Doorstop System Req | Risk Control | Verification |
|-----------------|-------------------|---------------------|--------------|--------------|
| **UR-EXPO-001** Precise Exposure Management | USR_OMNI-SERVER_005 | SYS_OMNI-SERVER_005, 012 | RISK_OMNI-SERVER_007 | VER_OMNI-SERVER_005, 007 |
| **UR-EXPO-002** Measurement Quality Validation | USR_OMNI-SERVER_006 | SYS_OMNI-SERVER_013 | RISK_OMNI-SERVER_008 | VER_OMNI-SERVER_008 |

### 3. Data Integrity and Security

| Professional Req | Doorstop User Req | Doorstop System Req | Risk Control | Verification |
|-----------------|-------------------|---------------------|--------------|--------------|
| **UR-DATA-001** Data Persistence and Fault Tolerance | USR_OMNI-SERVER_007 | SYS_OMNI-SERVER_006, 014 | RISK_OMNI-SERVER_005 | VER_OMNI-SERVER_004, 009 |
| **UR-DATA-002** Data Security and HIPAA Compliance | USR_OMNI-SERVER_008 | SYS_OMNI-SERVER_007, 009, 015 | RISK_OMNI-SERVER_004, 009 | VER_OMNI-SERVER_005, 010 |
| **UR-DATA-003** Complete Traceability | USR_OMNI-SERVER_009 | SYS_OMNI-SERVER_016 | - | VER_OMNI-SERVER_011 |

### 4. Calibration Management

| Professional Req | Doorstop User Req | Doorstop System Req | Risk Control | Verification |
|-----------------|-------------------|---------------------|--------------|--------------|
| **UR-CAL-001** Daily Calibration Enforcement | USR_OMNI-SERVER_002 | SYS_OMNI-SERVER_004 | - | VER_OMNI-SERVER_002, 004 |
| **UR-CAL-002** Calibration Diagnostics | USR_OMNI-SERVER_010 | SYS_OMNI-SERVER_017 | - | - |

### 5. User Interface and Usability

| Professional Req | Doorstop User Req | Doorstop System Req | Risk Control | Verification |
|-----------------|-------------------|---------------------|--------------|--------------|
| **UR-UI-001** Operator Interface Requirements | USR_OMNI-SERVER_011 | SYS_OMNI-SERVER_008, 022 | - | VER_OMNI-SERVER_002 |
| **UR-UI-002** Clinician Interface Requirements | USR_OMNI-SERVER_012 | SYS_OMNI-SERVER_008, 025 | - | - |

### 6. System Reliability

| Professional Req | Doorstop User Req | Doorstop System Req | Risk Control | Verification |
|-----------------|-------------------|---------------------|--------------|--------------|
| **UR-REL-001** Availability and Recovery | USR_OMNI-SERVER_013 | SYS_OMNI-SERVER_018 | - | - |
| **UR-REL-002** Diagnostic and Troubleshooting Support | USR_OMNI-SERVER_014 | SYS_OMNI-SERVER_019 | - | - |

### 7. Multi-User and Access Control

| Professional Req | Doorstop User Req | Doorstop System Req | Risk Control | Verification |
|-----------------|-------------------|---------------------|--------------|--------------|
| **UR-USER-001** Session Management | USR_OMNI-SERVER_015 | SYS_OMNI-SERVER_020 | - | - |
| **UR-USER-002** Role-Based Access Control | USR_OMNI-SERVER_016 | SYS_OMNI-SERVER_009, 021 | RISK_OMNI-SERVER_009 | VER_OMNI-SERVER_005 |

### 8. Error Handling and Recovery

| Professional Req | Doorstop User Req | Doorstop System Req | Risk Control | Verification |
|-----------------|-------------------|---------------------|--------------|--------------|
| **UR-ERR-001** Operator-Facing Error Messages | USR_OMNI-SERVER_017 | SYS_OMNI-SERVER_022 | - | - |
| **UR-ERR-002** Technical Diagnostic Information | USR_OMNI-SERVER_018 | SYS_OMNI-SERVER_019, 022 | - | - |

### 9. Software Update Management

| Professional Req | Doorstop User Req | Doorstop System Req | Risk Control | Verification |
|-----------------|-------------------|---------------------|--------------|--------------|
| **UR-UPD-001** Update Deployment Control | USR_OMNI-SERVER_019 | SYS_OMNI-SERVER_010, 023 | RISK_OMNI-SERVER_004, 011 | - |
| **UR-UPD-002** Update Impact Transparency | USR_OMNI-SERVER_020 | SYS_OMNI-SERVER_023 | - | - |

### 10. Regulatory Compliance and Audit Support

| Professional Req | Doorstop User Req | Doorstop System Req | Risk Control | Verification |
|-----------------|-------------------|---------------------|--------------|--------------|
| **UR-AUDIT-001** Comprehensive Audit Trail | USR_OMNI-SERVER_021 | SYS_OMNI-SERVER_007, 024 | RISK_OMNI-SERVER_010 | VER_OMNI-SERVER_004, 005, 012 |
| **UR-AUDIT-002** Quality Improvement Analytics | USR_OMNI-SERVER_022 | SYS_OMNI-SERVER_025 | - | - |

### 11. Network and Cloud Integration

| Professional Req | Doorstop User Req | Doorstop System Req | Risk Control | Verification |
|-----------------|-------------------|---------------------|--------------|--------------|
| **UR-NET-001** Offline Operation Capability | USR_OMNI-SERVER_023 | SYS_OMNI-SERVER_026 | - | VER_OMNI-SERVER_013 |
| **UR-NET-002** Network Management | USR_OMNI-SERVER_024 | SYS_OMNI-SERVER_026 | - | - |

### 12. Performance Requirements

| Professional Req | Doorstop User Req | Doorstop System Req | Risk Control | Verification |
|-----------------|-------------------|---------------------|--------------|--------------|
| **UR-PERF-001** Responsiveness | USR_OMNI-SERVER_025 | SYS_OMNI-SERVER_027 | - | VER_OMNI-SERVER_014 |
| **UR-PERF-002** Capacity and Scalability | USR_OMNI-SERVER_026 | - | - | - |

### 13. Training and Documentation

| Professional Req | Doorstop User Req | Doorstop System Req | Risk Control | Verification |
|-----------------|-------------------|---------------------|--------------|--------------|
| **UR-TRAIN-001** User Training Requirements | USR_OMNI-SERVER_027 | - | - | - |
| **UR-TRAIN-002** Documentation Requirements | USR_OMNI-SERVER_028 | - | - | - |

---

## Requirements Coverage Summary

### User Requirements (USR_OMNI-SERVER_xxx)
- **Existing:** 3 requirements (001-003)
- **Added:** 25 requirements (004-028)
- **Total:** 28 user requirements

### System Requirements (SYS_OMNI-SERVER_xxx)
- **Existing:** 10 requirements (001-010)
- **Added:** 17 requirements (011-027)
- **Total:** 27 system requirements

### Verification Requirements (VER_OMNI-SERVER_xxx)
- **Existing:** 5 requirements (001-005)
- **Added:** 9 requirements (006-014)
- **Total:** 14 verification requirements

### Risk Control Requirements (RISK_OMNI-SERVER_xxx)
- **Existing:** 5 requirements (001-005)
- **Added:** 6 requirements (006-011)
- **Total:** 11 risk control requirements

---

## Implementation Priority

### CRITICAL Priority (Must Implement First)
1. **Safety Controls:** USR_OMNI-SERVER_001, 004, 005 → SYS_OMNI-SERVER_001-003, 011, 012
2. **Data Integrity:** USR_OMNI-SERVER_007, 008 → SYS_OMNI-SERVER_014, 015
3. **Calibration Enforcement:** USR_OMNI-SERVER_002 → SYS_OMNI-SERVER_004
4. **Audit Trail:** USR_OMNI-SERVER_021 → SYS_OMNI-SERVER_024

### HIGH Priority (Implement Next)
1. **User Interface:** USR_OMNI-SERVER_011, 012 → SYS_OMNI-SERVER_008, 022, 025
2. **Access Control:** USR_OMNI-SERVER_015, 016 → SYS_OMNI-SERVER_020, 021
3. **Error Handling:** USR_OMNI-SERVER_017, 018 → SYS_OMNI-SERVER_019, 022
4. **Performance:** USR_OMNI-SERVER_025 → SYS_OMNI-SERVER_027

### MEDIUM Priority (Implement as Resources Allow)
1. **Network Management:** USR_OMNI-SERVER_023, 024 → SYS_OMNI-SERVER_026
2. **Update Management:** USR_OMNI-SERVER_019, 020 → SYS_OMNI-SERVER_023
3. **Analytics:** USR_OMNI-SERVER_022 → SYS_OMNI-SERVER_025
4. **Diagnostics:** USR_OMNI-SERVER_010, 014 → SYS_OMNI-SERVER_017, 019

---

## Gap Analysis

### Requirements with Full Traceability
✅ Safety and radiation protection  
✅ Exposure control  
✅ Data security and HIPAA compliance  
✅ Calibration management  
✅ Audit trail  

### Requirements Needing Additional Implementation Details
⚠️ Training and documentation (no system requirements defined yet - these are process/deliverable requirements)  
⚠️ Capacity and scalability (needs system resource planning)  
⚠️ Some UI requirements (need frontend application requirements)

### Recommendations
1. **Frontend Application Requirements:** Many UI requirements (USR_OMNI-SERVER_011, 012) will be implemented in a separate frontend application (e.g., omniscan-orchestrator or dedicated UI component). Consider creating separate requirement sets for frontend components.

2. **Cross-Component Requirements:** Some requirements span multiple components:
   - Session management (hardware server + orchestrator)
   - User interface (frontend application)
   - Analytics (potentially separate analytics service)

3. **Process Requirements:** Training and documentation requirements (USR_OMNI-SERVER_027, 028) are process requirements rather than software requirements. Consider creating separate documentation plan and training plan documents.

---

## Next Steps

1. ✅ **Complete:** User requirements created (USR_OMNI-SERVER_001-028)
2. ✅ **Complete:** System requirements created for hardware server scope (SYS_OMNI-SERVER_001-027)
3. ✅ **Complete:** Verification requirements created (VER_OMNI-SERVER_001-014)
4. ✅ **Complete:** Risk control requirements created (RISK_OMNI-SERVER_001-011)
5. **TODO:** Review and validate with stakeholders
6. **TODO:** Create corresponding requirements for frontend components
7. **TODO:** Update Doorstop .doorstop.yml configuration files
8. **TODO:** Generate Doorstop requirement reports (HTML/PDF)
9. **TODO:** Link to design documents and test cases as development proceeds

---

## Document Control

**Created:** October 25, 2025  
**Author:** Omniscan Development Team  
**Status:** Initial Draft  
**Review Required:** Yes

---

**End of Traceability Matrix**
