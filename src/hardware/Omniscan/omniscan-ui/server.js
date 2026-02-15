/**
 * REST API Proxy Server for Omniscan UI
 * Proxies HTTP/REST requests from the browser to the orchestrator's REST API
 */

import express from 'express';
import cors from 'cors';

const app = express();
const PORT = 3001;
const ORCHESTRATOR_URL = 'http://localhost:8080';

// Middleware
app.use(cors());
app.use(express.json());

let connected = false;

// Check orchestrator connection (called on-demand, not periodically)
async function checkConnection() {
  try {
    // Use debug endpoint that doesn't require auth
    const response = await fetch(`${ORCHESTRATOR_URL}/api/debug/status`, { 
      method: 'GET',
      signal: AbortSignal.timeout(3000)  // 3 second timeout
    });
    connected = response.ok;
    return connected;
  } catch (error) {
    connected = false;
    return false;
  }
}

// Initialize connection check on startup
checkConnection().then(status => {
  console.log(`Orchestrator connection: ${status ? 'connected' : 'disconnected'}`);
});

// No periodic checking - connection status is checked on-demand via /api/connection/status

// Simple authentication store (in-memory for now)
const sessions = new Map();

/**
 * POST /api/auth/login
 * Authenticate user and create session via orchestrator
 */
app.post('/api/auth/login', async (req, res) => {
  const { username, password } = req.body;

  try {
    const response = await fetch(`${ORCHESTRATOR_URL}/api/auth/login`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ username, password })
    });

    const data = await response.json();

    if (data.success && data.session_id) {
      // Store session mapping
      sessions.set(data.session_id, {
        sessionId: data.session_id,
        username: data.user_id,
        role: data.role,
        loginTime: new Date().toISOString()
      });

      // Return formatted response for UI
      res.json({
        success: true,
        data: {
          sessionId: data.session_id,
          user: {
            username: data.user_id,
            role: data.role,
            permissions: ['read', 'write', 'admin']
          }
        }
      });
    } else {
      res.status(401).json({
        success: false,
        error: data.error || 'Authentication failed'
      });
    }
  } catch (error) {
    console.error('Login failed:', error);
    res.status(503).json({
      success: false,
      error: 'Cannot connect to orchestrator'
    });
  }
});

/**
 * POST /api/auth/logout
 * End user session
 */
app.post('/api/auth/logout', async (req, res) => {
  const sessionId = req.headers['x-session-id'];
  
  if (!sessionId || !sessions.has(sessionId)) {
    return res.status(401).json({ success: false, error: 'Invalid session' });
  }

  try {
    // Call orchestrator logout
    await fetch(`${ORCHESTRATOR_URL}/api/auth/logout`, {
      method: 'POST',
      headers: { 'x-session-id': sessionId }
    });

    // Remove local session
    sessions.delete(sessionId);
    res.json({ success: true });
  } catch (error) {
    // Even if orchestrator call fails, clean up local session
    sessions.delete(sessionId);
    res.json({ success: true });
  }
});

/**
 * GET /api/health
 * Check orchestrator health status (requires authentication)
 */
app.get('/api/health', async (req, res) => {
  const sessionId = req.headers['x-session-id'];
  
  if (!sessionId) {
    return res.status(401).json({
      success: false,
      error: 'Unauthorized'
    });
  }

  try {
    const response = await fetch(`${ORCHESTRATOR_URL}/api/health`, {
      headers: { 'x-session-id': sessionId }
    });
    
    if (!response.ok) {
      return res.status(503).json({
        success: false,
        error: 'Orchestrator not responding'
      });
    }

    const data = await response.json();
    
    // Transform snake_case to camelCase for UI
    const transformed = {
      state: data.state,
      interlocks: {
        overall_safe: data.interlocks.overall_safe,
        keySwitch: data.interlocks.key_switch,
        enableButton: data.interlocks.enable_button,
        doorSensor: data.interlocks.door_closed,
        emergencyStop: data.interlocks.emergency_stop,
        beamWatchdog: data.interlocks.radiation_safe,
        coolingOk: data.interlocks.cooling_ok,
        powerOk: data.interlocks.power_ok
      },
      calibration: data.calibration ? {
        id: data.calibration.id,
        timestamp: data.calibration.timestamp,
        valid: data.calibration.valid,
        expiresAt: data.calibration.expires_at,
        distanceCheck: data.calibration.distance_check,
        snrThreshold: data.calibration.snr_threshold,
        parameters: {}
      } : null,
      uptime: data.uptime || 0,
      cloudConnected: data.cloud_connected || false,
      lastHeartbeat: data.last_heartbeat || new Date().toISOString()
    };
    
    res.json({ success: true, data: transformed });
  } catch (error) {
    console.error('Health check failed:', error);
    res.status(503).json({
      success: false,
      error: 'Not connected to orchestrator'
    });
  }
});

/**
 * GET /api/connection/status
 * Check if UI is connected to orchestrator
 */
app.get('/api/connection/status', (req, res) => {
  res.json({
    success: true,
    data: {
      connected,
      orchestratorAddress: 'localhost:8080'
    }
  });
});

/**
 * POST /api/connection/connect
 * Attempt to connect to orchestrator
 */
app.post('/api/connection/connect', async (req, res) => {
  const isConnected = await checkConnection();

  res.json({
    success: isConnected,
    data: {
      connected: isConnected,
      message: isConnected ? 'Connected successfully' : 'Connection failed'
    }
  });
});

/**
 * GET /api/state
 * Get compact system state (for polling)
 */
app.get('/api/state', async (req, res) => {
  const sessionId = req.headers['x-session-id'];
  
  if (!sessionId) {
    return res.status(401).json({
      success: false,
      error: 'Unauthorized'
    });
  }

  try {
    const response = await fetch(`${ORCHESTRATOR_URL}/api/health`, {
      headers: { 'x-session-id': sessionId }
    });
    
    if (!response.ok) {
      const errorData = await response.json().catch(() => ({}));
      console.error('Health check failed:', response.status, errorData);
      return res.status(response.status).json({
        success: false,
        error: errorData.error || 'Failed to get system state'
      });
    }

    const health = await response.json();
    
    // Validate required fields exist
    if (!health.interlocks) {
      console.error('Invalid health data structure:', health);
      return res.status(500).json({
        success: false,
        error: 'Invalid health data from orchestrator'
      });
    }
    
    // Transform health data to compact state format expected by UI
    const state = {
      system_state: health.state,  // Note: orchestrator uses 'state' not 'system_state'
      devices: {
        pdu: {
          powered: health.pdu?.powered ?? false,
          status: health.pdu?.status ?? 'Off'
        },
        gpio: {
          powered: health.gpio?.powered ?? false,
          status: health.gpio?.status ?? 'Off'
        },
        detector: {
          powered: health.detector?.powered ?? false,
          status: health.detector?.status ?? 'OFF'
        },
        motion: {
          powered: health.motion?.powered ?? false,
          status: health.motion?.status ?? 'OFF'
        }
      },
      interlocks: {
        overall_safe: health.interlocks.overall_safe,
        key_switch: health.interlocks.key_switch,
        enable_button: health.interlocks.enable_button,
        door_closed: health.interlocks.door_closed,
        emergency_stop: health.interlocks.emergency_stop,
        cooling_ok: health.interlocks.cooling_ok,
        power_ok: health.interlocks.power_ok,
        radiation_safe: health.interlocks.radiation_safe ?? true
      },
      timestamp: health.last_heartbeat
    };
    
    res.json({ success: true, data: state });
  } catch (error) {
    console.error('State check failed:', error);
    res.status(503).json({
      success: false,
      error: 'Cannot connect to orchestrator'
    });
  }
});

/**
 * GET /api/gpio/enable-button
 * Get enable button status from orchestrator
 */
app.get('/api/gpio/enable-button', async (req, res) => {
  const sessionId = req.headers['x-session-id'];
  
  if (!sessionId) {
    return res.status(401).json({
      success: false,
      error: 'Unauthorized'
    });
  }

  try {
    const response = await fetch(`${ORCHESTRATOR_URL}/api/gpio/enable-button`, {
      headers: { 'x-session-id': sessionId }
    });
    
    if (!response.ok) {
      return res.status(response.status).json({
        success: false,
        error: 'Failed to get enable button status'
      });
    }

    const data = await response.json();
    res.json(data);
  } catch (error) {
    console.error('Enable button check failed:', error);
    res.status(503).json({
      success: false,
      error: 'Cannot connect to orchestrator'
    });
  }
});

// Middleware to check session
function requireSession(req, res, next) {
  const sessionId = req.headers['x-session-id'];
  if (!sessionId || !sessions.has(sessionId)) {
    return res.status(401).json({
      success: false,
      error: 'Unauthorized - please login'
    });
  }
  req.session = sessions.get(sessionId);
  next();
}

// Helper function to generate session ID
function generateSessionId() {
  return `session_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
}

/**
 * Catch-all proxy for any other /api requests
 * Forward them directly to orchestrator with session
 */
app.use('/api', async (req, res, next) => {
  // Skip if already handled by specific routes above
  if (res.headersSent) {
    return next();
  }
  const sessionId = req.headers['x-session-id'];
  
  // Most endpoints require auth, but connection/status doesn't
  if (!req.path.includes('/connection/') && !sessionId) {
    return res.status(401).json({
      success: false,
      error: 'Unauthorized'
    });
  }
  
  try {
    // req.path doesn't include /api when using app.use('/api', ...)
    // so we need to add it back
    const url = `${ORCHESTRATOR_URL}/api${req.path}`;
    const headers = {
      'Content-Type': 'application/json'
    };
    
    if (sessionId) {
      headers['x-session-id'] = sessionId;
    }
    
    const options = {
      method: req.method,
      headers
    };
    
    // Only add body for POST/PUT/PATCH requests with actual content
    if (['POST', 'PUT', 'PATCH'].includes(req.method) && req.body && Object.keys(req.body).length > 0) {
      options.body = JSON.stringify(req.body);
    }
    
    console.log(`Proxying ${req.method} /api${req.path} to ${url}`);
    const response = await fetch(url, options);
    
    const contentType = response.headers.get('content-type');
    if (contentType && contentType.includes('application/json')) {
      const data = await response.json();
      res.status(response.status).json(data);
    } else {
      const text = await response.text();
      res.status(response.status).send(text);
    }
  } catch (error) {
    console.error(`Proxy error for /api${req.path}:`, error.message);
    console.error('Full error:', error);
    res.status(503).json({
      success: false,
      error: 'Cannot connect to orchestrator: ' + error.message
    });
  }
});

// Start server
app.listen(PORT, () => {
  console.log(`Omniscan UI API Server running on http://localhost:${PORT}`);
  console.log(`Proxying to orchestrator at ${ORCHESTRATOR_URL}`);
});
