/**
 * Tests for REST API Proxy Server (server.js)
 * Tests all endpoints and their interactions with the orchestrator
 */

import { describe, it, expect, beforeAll, afterAll, beforeEach, vi } from 'vitest';
import request from 'supertest';
import express from 'express';
import cors from 'cors';

// Mock fetch globally
global.fetch = vi.fn();

// Set up the app (same as server.js but without listening)
const app = express();
const ORCHESTRATOR_URL = 'http://localhost:8081';

app.use(cors());
app.use(express.json());

let connected = false;
const sessions = new Map();

// Replicate server endpoints for testing
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
      sessions.set(data.session_id, {
        sessionId: data.session_id,
        username: data.user_id,
        role: data.role,
        loginTime: new Date().toISOString()
      });

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
    res.status(503).json({
      success: false,
      error: 'Cannot connect to orchestrator'
    });
  }
});

app.post('/api/auth/logout', async (req, res) => {
  const sessionId = req.headers['x-session-id'];
  
  if (!sessionId || !sessions.has(sessionId)) {
    return res.status(401).json({ success: false, error: 'Invalid session' });
  }

  try {
    await fetch(`${ORCHESTRATOR_URL}/api/auth/logout`, {
      method: 'POST',
      headers: { 'x-session-id': sessionId }
    });

    sessions.delete(sessionId);
    res.json({ success: true });
  } catch (error) {
    sessions.delete(sessionId);
    res.json({ success: true });
  }
});

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
    res.status(503).json({
      success: false,
      error: 'Not connected to orchestrator'
    });
  }
});

app.get('/api/connection/status', (req, res) => {
  res.json({
    success: true,
    data: {
      connected,
      orchestratorAddress: 'localhost:8081'
    }
  });
});

app.post('/api/connection/connect', async (req, res) => {
  try {
    const response = await fetch(`${ORCHESTRATOR_URL}/docs`, { 
      method: 'HEAD'
    });
    connected = response.ok;
  } catch (error) {
    connected = false;
  }

  res.json({
    success: connected,
    data: {
      connected: connected,
      message: connected ? 'Connected successfully' : 'Connection failed'
    }
  });
});

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
      return res.status(response.status).json({
        success: false,
        error: errorData.error || 'Failed to get system state'
      });
    }

    const health = await response.json();
    
    if (!health.interlocks) {
      return res.status(500).json({
        success: false,
        error: 'Invalid health data from orchestrator'
      });
    }
    
    const state = {
      system_state: health.state,
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
        radiation_safe: health.interlocks.radiation_safe,
        cooling_ok: health.interlocks.cooling_ok,
        power_ok: health.interlocks.power_ok
      },
      timestamp: health.last_heartbeat
    };
    
    res.json({ success: true, data: state });
  } catch (error) {
    res.status(503).json({
      success: false,
      error: 'Cannot connect to orchestrator'
    });
  }
});

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
    res.status(503).json({
      success: false,
      error: 'Cannot connect to orchestrator'
    });
  }
});

// Catch-all proxy
app.use('/api', async (req, res, next) => {
  if (res.headersSent) {
    return next();
  }
  const sessionId = req.headers['x-session-id'];
  
  if (!req.path.includes('/connection/') && !sessionId) {
    return res.status(401).json({
      success: false,
      error: 'Unauthorized'
    });
  }
  
  try {
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
    
    if (['POST', 'PUT', 'PATCH'].includes(req.method) && req.body && Object.keys(req.body).length > 0) {
      options.body = JSON.stringify(req.body);
    }
    
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
    res.status(503).json({
      success: false,
      error: 'Cannot connect to orchestrator: ' + error.message
    });
  }
});

describe('Server API Tests', () => {
  beforeEach(() => {
    vi.clearAllMocks();
    sessions.clear();
    connected = false;
  });

  describe('POST /api/auth/login', () => {
    it('should successfully login with valid credentials', async () => {
      const mockResponse = {
        success: true,
        session_id: 'test-session-123',
        user_id: 'admin',
        role: 'admin'
      };

      global.fetch.mockResolvedValueOnce({
        json: async () => mockResponse
      });

      const response = await request(app)
        .post('/api/auth/login')
        .send({ username: 'admin', password: 'admin123' });

      expect(response.status).toBe(200);
      expect(response.body.success).toBe(true);
      expect(response.body.data.sessionId).toBe('test-session-123');
      expect(response.body.data.user.username).toBe('admin');
      expect(response.body.data.user.role).toBe('admin');
    });

    it('should reject login with invalid credentials', async () => {
      const mockResponse = {
        success: false,
        error: 'Invalid credentials'
      };

      global.fetch.mockResolvedValueOnce({
        json: async () => mockResponse
      });

      const response = await request(app)
        .post('/api/auth/login')
        .send({ username: 'invalid', password: 'wrong' });

      expect(response.status).toBe(401);
      expect(response.body.success).toBe(false);
      expect(response.body.error).toBeTruthy();
    });

    it('should handle orchestrator connection failure', async () => {
      global.fetch.mockRejectedValueOnce(new Error('Connection refused'));

      const response = await request(app)
        .post('/api/auth/login')
        .send({ username: 'admin', password: 'admin123' });

      expect(response.status).toBe(503);
      expect(response.body.success).toBe(false);
      expect(response.body.error).toBe('Cannot connect to orchestrator');
    });
  });

  describe('POST /api/auth/logout', () => {
    it('should successfully logout with valid session', async () => {
      const sessionId = 'test-session-123';
      sessions.set(sessionId, { sessionId, username: 'admin' });

      global.fetch.mockResolvedValueOnce({
        json: async () => ({ success: true })
      });

      const response = await request(app)
        .post('/api/auth/logout')
        .set('x-session-id', sessionId);

      expect(response.status).toBe(200);
      expect(response.body.success).toBe(true);
      expect(sessions.has(sessionId)).toBe(false);
    });

    it('should reject logout with invalid session', async () => {
      const response = await request(app)
        .post('/api/auth/logout')
        .set('x-session-id', 'invalid-session');

      expect(response.status).toBe(401);
      expect(response.body.success).toBe(false);
    });

    it('should clean up session even if orchestrator call fails', async () => {
      const sessionId = 'test-session-123';
      sessions.set(sessionId, { sessionId, username: 'admin' });

      global.fetch.mockRejectedValueOnce(new Error('Connection refused'));

      const response = await request(app)
        .post('/api/auth/logout')
        .set('x-session-id', sessionId);

      expect(response.status).toBe(200);
      expect(response.body.success).toBe(true);
      expect(sessions.has(sessionId)).toBe(false);
    });
  });

  describe('GET /api/health', () => {
    it('should return health data with valid session', async () => {
      const mockHealth = {
        state: 'IDLE',
        interlocks: {
          overall_safe: true,
          key_switch: true,
          enable_button: true,
          door_closed: true,
          emergency_stop: false,
          radiation_safe: true,
          cooling_ok: true,
          power_ok: true
        },
        calibration: {
          id: 'cal_001',
          timestamp: '2025-01-24T12:00:00Z',
          valid: true,
          expires_at: '2025-01-25T12:00:00Z',
          distance_check: true,
          snr_threshold: 20.0
        },
        uptime: 3600,
        cloud_connected: true,
        last_heartbeat: '2025-01-24T13:00:00Z'
      };

      global.fetch.mockResolvedValueOnce({
        ok: true,
        json: async () => mockHealth
      });

      const response = await request(app)
        .get('/api/health')
        .set('x-session-id', 'test-session');

      expect(response.status).toBe(200);
      expect(response.body.success).toBe(true);
      expect(response.body.data.state).toBe('IDLE');
      expect(response.body.data.interlocks.keySwitch).toBe(true);
      expect(response.body.data.calibration.id).toBe('cal_001');
    });

    it('should reject request without session', async () => {
      const response = await request(app).get('/api/health');

      expect(response.status).toBe(401);
      expect(response.body.success).toBe(false);
      expect(response.body.error).toBe('Unauthorized');
    });

    it('should handle orchestrator not responding', async () => {
      global.fetch.mockResolvedValueOnce({
        ok: false,
        json: async () => ({})
      });

      const response = await request(app)
        .get('/api/health')
        .set('x-session-id', 'test-session');

      expect(response.status).toBe(503);
      expect(response.body.success).toBe(false);
    });
  });

  describe('GET /api/connection/status', () => {
    it('should return connection status', async () => {
      const response = await request(app).get('/api/connection/status');

      expect(response.status).toBe(200);
      expect(response.body.success).toBe(true);
      expect(response.body.data).toHaveProperty('connected');
      expect(response.body.data.orchestratorAddress).toBe('localhost:8081');
    });
  });

  describe('POST /api/connection/connect', () => {
    it('should successfully connect to orchestrator', async () => {
      global.fetch.mockResolvedValueOnce({
        ok: true
      });

      const response = await request(app).post('/api/connection/connect');

      expect(response.status).toBe(200);
      expect(response.body.success).toBe(true);
      expect(response.body.data.connected).toBe(true);
    });

    it('should fail to connect when orchestrator is down', async () => {
      global.fetch.mockRejectedValueOnce(new Error('Connection refused'));

      const response = await request(app).post('/api/connection/connect');

      expect(response.status).toBe(200);
      expect(response.body.success).toBe(false);
      expect(response.body.data.connected).toBe(false);
    });
  });

  describe('GET /api/state', () => {
    it('should return system state with valid session', async () => {
      const mockHealth = {
        state: 'SCANNING',
        interlocks: {
          overall_safe: true,
          key_switch: true,
          enable_button: true,
          door_closed: true,
          emergency_stop: false,
          radiation_safe: true,
          cooling_ok: true,
          power_ok: true
        },
        last_heartbeat: '2025-01-24T13:00:00Z'
      };

      global.fetch.mockResolvedValueOnce({
        ok: true,
        json: async () => mockHealth
      });

      const response = await request(app)
        .get('/api/state')
        .set('x-session-id', 'test-session');

      expect(response.status).toBe(200);
      expect(response.body.success).toBe(true);
      expect(response.body.data.system_state).toBe('SCANNING');
      expect(response.body.data.interlocks).toBeDefined();
      expect(response.body.data.devices).toBeDefined();
    });

    it('should reject request without session', async () => {
      const response = await request(app).get('/api/state');

      expect(response.status).toBe(401);
      expect(response.body.success).toBe(false);
    });

    it('should handle invalid health data', async () => {
      global.fetch.mockResolvedValueOnce({
        ok: true,
        json: async () => ({ state: 'IDLE' }) // Missing interlocks
      });

      const response = await request(app)
        .get('/api/state')
        .set('x-session-id', 'test-session');

      expect(response.status).toBe(500);
      expect(response.body.success).toBe(false);
      expect(response.body.error).toBe('Invalid health data from orchestrator');
    });
  });

  describe('GET /api/gpio/enable-button', () => {
    it('should return enable button status with valid session', async () => {
      const mockData = {
        success: true,
        data: { pressed: true, timestamp: '2025-01-24T13:00:00Z' }
      };

      global.fetch.mockResolvedValueOnce({
        ok: true,
        json: async () => mockData
      });

      const response = await request(app)
        .get('/api/gpio/enable-button')
        .set('x-session-id', 'test-session');

      expect(response.status).toBe(200);
      expect(response.body.success).toBe(true);
    });

    it('should reject request without session', async () => {
      const response = await request(app).get('/api/gpio/enable-button');

      expect(response.status).toBe(401);
      expect(response.body.success).toBe(false);
    });
  });

  describe('Catch-all proxy /api/*', () => {
    it('should proxy authenticated requests to orchestrator', async () => {
      const mockData = { success: true, data: { test: 'data' } };

      global.fetch.mockResolvedValueOnce({
        status: 200,
        headers: {
          get: (name) => name === 'content-type' ? 'application/json' : null
        },
        json: async () => mockData
      });

      const response = await request(app)
        .get('/api/custom/endpoint')
        .set('x-session-id', 'test-session');

      expect(response.status).toBe(200);
      expect(response.body).toEqual(mockData);
    });

    it('should reject unauthenticated requests', async () => {
      const response = await request(app).get('/api/custom/endpoint');

      expect(response.status).toBe(401);
      expect(response.body.success).toBe(false);
    });

    it('should handle orchestrator connection failure', async () => {
      global.fetch.mockRejectedValueOnce(new Error('Network error'));

      const response = await request(app)
        .get('/api/custom/endpoint')
        .set('x-session-id', 'test-session');

      expect(response.status).toBe(503);
      expect(response.body.success).toBe(false);
    });
  });
});
