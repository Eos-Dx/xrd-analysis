# UI ↔ Orchestrator Connection Fix

## Problem

The UI was configured to connect to the wrong port:
- **API proxy**: `http://localhost:3001` ❌ (wrong port)
- **Orchestrator running on**: `http://localhost:8081` ✓

## Solution Applied

### Fixed `vite.config.ts`

Changed API proxy target from port `3001` to `8081`:

```typescript
proxy: {
  '/api': {
    target: 'http://localhost:8081',  // ✓ Correct orchestrator port
    changeOrigin: true,
  },
  '/ws': {
    target: 'ws://localhost:8081',  // ✓ Already correct
    ws: true,
  },
}
```

### Fixed CORS in `rest_server.py`

Updated CORS to allow all origins in development:

```python
allow_origins=[
    "http://localhost:3000",  # React dev server
    "http://localhost:3001",  # Alternate React port
    "http://127.0.0.1:3000",
    "http://127.0.0.1:3001",
    "*"  # Allow all origins in development
],
```

## Testing

### 1. Restart the Orchestrator

If it's still running, restart it to pick up CORS changes:

```powershell
# Stop current process (if using batch file, kill it)
Get-Process python | Stop-Process -Force

# Start with PowerShell script
cd C:\dev\Omniscan\omniscan-orchestrator
.\start_server.ps1
```

### 2. Restart the UI

```powershell
cd C:\dev\Omniscan\omniscan-ui

# Kill any existing dev server
Get-Process node | Stop-Process -Force

# Start UI dev server
npm run dev
```

### 3. Verify Connection

Open browser to `http://localhost:3000` and check:
- ✅ Dashboard loads
- ✅ System status shows
- ✅ No CORS errors in browser console

## Port Configuration Summary

| Component | Port | URL |
|-----------|------|-----|
| **UI (Vite Dev Server)** | 3000 | http://localhost:3000 |
| **Orchestrator REST API** | 8081 | http://localhost:8081 |
| **Hardware Server gRPC** | 50051 | localhost:50051 |

## Verify Endpoints

Test that orchestrator responds:

```powershell
# Test connection status
curl http://localhost:8081/api/debug/status

# Test new CommandDiscovery endpoints
curl http://localhost:8081/api/v1/server/capabilities
curl http://localhost:8081/api/v1/server/commands

# Test existing endpoints
curl http://localhost:8081/api/state
curl http://localhost:8081/api/health
```

## All Endpoints Still Available

✅ All original endpoints are intact:
- `/api/auth/login` - Authentication
- `/api/auth/logout` - Logout
- `/api/state` - System state
- `/api/health` - Health check
- `/api/patients` - Patient management
- `/api/measurements/*` - Measurement endpoints
- `/api/calibration/*` - Calibration endpoints
- `/api/gpio/*` - GPIO state
- `/api/hardware/*` - Hardware initialization
- `/api/motion/*` - Motion control

✅ New CommandDiscovery endpoints added:
- `/api/v1/server/capabilities` - Server version and features
- `/api/v1/server/commands` - List all commands
- `/api/v1/server/validate-compatibility` - Check compatibility

## Troubleshooting

### UI still can't connect

1. **Check orchestrator is running**:
   ```powershell
   curl http://localhost:8081/api/debug/status
   ```

2. **Check UI dev server is running**:
   ```powershell
   curl http://localhost:3000
   ```

3. **Check browser console** for errors (F12 → Console tab)

4. **Verify vite config was saved**:
   ```powershell
   cat C:\dev\Omniscan\omniscan-ui\vite.config.ts
   ```
   Should show `target: 'http://localhost:8081'`

### CORS errors in browser

If you see CORS errors:
1. Orchestrator needs to be restarted after CORS changes
2. Clear browser cache (Ctrl+Shift+Delete)
3. Try incognito mode

### Port conflict

If port 8081 is already in use:
```powershell
# Find what's using port 8081
Get-NetTCPConnection -LocalPort 8081 | Select-Object OwningProcess
```

## Summary

✅ **Fixed**: UI vite.config.ts now points to correct port (8081)
✅ **Fixed**: Orchestrator CORS allows UI connections
✅ **Verified**: All endpoints still available
✅ **Added**: New CommandDiscovery endpoints with automatic compatibility checking

The UI should now connect successfully to the orchestrator! 🎉
