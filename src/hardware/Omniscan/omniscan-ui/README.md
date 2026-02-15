# Omniscan UI

React-based web UI for the Omniscan Medical XRD Diagnostic System.

## Overview

This UI provides a comprehensive interface for operating the Omniscan Medical XRD Diagnostic System, with focus on safety, compliance, and ease of use.

### Core Features

- **Dashboard**: System status overview with safety interlocks, calibration status, and quick measurement controls
- **Measurement**: Run XRD measurements with sample tracking and view measurement history
- **Calibration**: Perform async daily calibration with comprehensive QC reporting
- **Real-time Updates**: WebSocket connection for live system state updates
- **Role-Based Access**: Different views for Operators, Engineers, and Administrators
- **Patient Management**: Patient registration, search by MRN, and measurement linking
- **Hardware Control**: Device initialization, diagnostics, and monitoring

## Tech Stack

- **React 18** with TypeScript
- **Vite** for fast development and building
- **Zustand** for state management
- **React Router** for navigation
- **Tailwind CSS** for styling
- **date-fns** for date formatting
- **Recharts** for data visualization

## Architecture

### Key Features Aligned with User Expectations

Based on `USER_EXPECTATIONS.md`:

1. **Safety First**
   - Real-time safety interlock status display
   - Clear visual indication of system state
   - Emergency abort controls always accessible
   - Large, clearly visible status indicators

2. **Data Integrity**
   - All API calls include error handling
   - Offline capability supported (local buffering in orchestrator)
   - Cloud connection status always visible
   - Complete measurement traceability

3. **Clinical Reliability**
   - Simple 4-click measurement workflow
   - Plain language status messages ("Ready", "Running", "Calibration Required")
   - Daily calibration enforcement
   - Role-based access control

### Component Structure

```
src/
├── components/
│   ├── auth/           # Login and authentication
│   ├── calibration/    # Calibration workflow
│   ├── common/         # Reusable UI components
│   ├── layout/         # App layout and navigation
│   ├── measurement/    # Measurement controls and history
│   └── status/         # System status displays
├── pages/              # Route-level pages
├── services/           # API and WebSocket clients
├── store/              # Zustand state management
├── types/              # TypeScript type definitions
└── App.tsx             # Main application
```

## Getting Started

### Prerequisites

- Node.js 18+ and npm
- Access to Omniscan orchestrator backend (default: http://localhost:8080)

### Installation

```bash
# Install dependencies
npm install

# Start development server
npm run dev
```

The UI will be available at http://localhost:3000

### Build for Production

```bash
npm run build
```

Built files will be in the `dist/` directory.

## Development

### Type Checking

```bash
npm run typecheck
```

### Linting

```bash
npm run lint
```

## Configuration

### API Proxy

The Vite dev server proxies API requests to the orchestrator:

- `/api/*` → `http://localhost:8080/api/*`
- `/ws` → `ws://localhost:8080/ws`

Modify `vite.config.ts` to change backend URL.

### System States

The UI reflects these operational states from the hardware server:

- `IDLE` - Ready for measurements
- `PENDING_ARMED` - Arming sequence in progress
- `RUNNING` - Measurement active
- `STOPPING` - Shutdown in progress
- `SAFE` - Safe mode (fault detected)
- `CALIBRATION` - Calibration in progress
- `MAINTENANCE` - Maintenance mode (engineers only)
- `LOCKED` - Calibration expired, measurements blocked

## Key Workflows

### Measurement Workflow

1. Login with operator credentials
2. Verify calibration is valid (green badge in header)
3. Check safety interlocks (all must show "OK")
4. Navigate to Measurement page
5. Enter Sample ID
6. Set exposure duration
7. Click "Start Measurement"
8. Monitor progress in real-time
9. View results in history table

### Daily Calibration

1. Navigate to Calibration page
2. Insert calibration standard sample
3. Close safety door
4. Click "Start Daily Calibration"
5. System performs automatic validation
6. Review pass/fail status
7. Calibration valid for 24 hours

## Security

- User authentication required for all operations
- Role-based UI element visibility
- Session management with automatic timeout
- TLS encryption for all API traffic (in production)
- WebSocket uses same authentication context

## Compliance

This UI is part of an IEC 62304 Class B medical device system:

- Complete audit trail of user actions
- Traceability linking measurements to operators, calibrations, and device state
- Data integrity with immediate persistence
- HIPAA-compliant data handling
- FDA cybersecurity guidelines compliance

## Documentation

Detailed documentation is available in the following files:

- **[Architecture.md](Architecture.md)** - System architecture, component structure, data flow, and WebSocket events
- **[Implementation.md](Implementation.md)** - Current implementation status, completed features, and known issues
- **[Functional_Requirements.md](Functional_Requirements.md)** - Detailed functional requirements and acceptance criteria
- **[Future_Implementation.md](Future_Implementation.md)** - Planned enhancements and roadmap

## Quick Start

### Prerequisites

- Node.js 18+ and npm
- Access to Omniscan orchestrator backend (default: http://localhost:8081)

### Development

```bash
# Install dependencies
npm install

# Start development server (runs on http://localhost:3000)
npm run dev

# Type checking
npm run typecheck

# Linting
npm run lint

# Build for production
npm run build
```

## System Requirements

### Browser Support
- Chrome 90+ (recommended)
- Firefox 88+
- Safari 14+
- Edge 90+

### Minimum Screen Resolution
- Desktop: 1280x720 (primary use case)
- Tablet: 768x1024 (view-only)

## Compliance

This UI is part of an IEC 62304 Class B medical device system:

- **IEC 62304 Class B** - Medical device software lifecycle
- **HIPAA Compliant** - Patient data handling
- **FDA Cybersecurity** - Security guidelines compliance
- **Complete Audit Trail** - All user actions logged
- **Data Integrity** - Immediate persistence and traceability

## Key Workflows

### Measurement (4 Clicks)
1. Navigate to Measurement page
2. Enter Sample ID
3. Set exposure duration
4. Click "Start Measurement"

### Daily Calibration
1. Navigate to Calibration page
2. Insert calibration standard (LaB₆)
3. Close safety door
4. Click "Start Daily Calibration"
5. Review QC report when complete
6. Accept calibration if all checks pass

## Support & Contact

For technical support, contact the Omniscan development team.

## License

TBD
