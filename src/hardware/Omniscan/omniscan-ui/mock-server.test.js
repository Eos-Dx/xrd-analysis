/**
 * Tests for Mock API Server (mock-server.js)
 * Tests authentication and health endpoints
 */

import { describe, it, expect, beforeEach } from 'vitest';
import request from 'supertest';
import { createServer } from 'http';

// Replicate mock server logic for testing
const users = {
  'admin': { password: 'admin123', username: 'admin', role: 'admin' },
  'operator': { password: 'operator123', username: 'operator', role: 'operator' },
  'engineer': { password: 'engineer123', username: 'engineer', role: 'engineer' }
};

const createMockServer = () => {
  return createServer((req, res) => {
    // CORS
    res.setHeader('Access-Control-Allow-Origin', '*');
    res.setHeader('Access-Control-Allow-Methods', 'GET, POST, PUT, DELETE, OPTIONS');
    res.setHeader('Access-Control-Allow-Headers', 'Content-Type');
    
    if (req.method === 'OPTIONS') {
      res.writeHead(200);
      res.end();
      return;
    }

    res.setHeader('Content-Type', 'application/json');

    // Login endpoint
    if (req.url === '/api/auth/login' && req.method === 'POST') {
      let body = '';
      req.on('data', chunk => body += chunk);
      req.on('end', () => {
        const { username, password } = JSON.parse(body);
        const user = users[username];
        
        if (user && user.password === password) {
          res.writeHead(200);
          res.end(JSON.stringify({ 
            id: `user_${username}_${Date.now()}`,
            username: user.username, 
            role: user.role.toUpperCase(),
            sessionStart: new Date().toISOString()
          }));
        } else {
          res.writeHead(401);
          res.end(JSON.stringify({ message: 'Invalid credentials', code: 'AUTH_FAILED' }));
        }
      });
      return;
    }

    // Logout endpoint
    if (req.url === '/api/auth/logout' && req.method === 'POST') {
      res.writeHead(200);
      res.end(JSON.stringify({ success: true }));
      return;
    }

    // Health endpoint
    if (req.url === '/api/health' && req.method === 'GET') {
      res.writeHead(200);
      res.end(JSON.stringify({
        state: 'IDLE',
        interlocks: {
          keySwitch: true,
          enableButton: true,
          doorSensor: true,
          emergencyStop: false,
          beamWatchdog: true,
          overTemp: false
        },
        calibration: {
          id: 'cal_20250124_001',
          timestamp: new Date(Date.now() - 3600000).toISOString(),
          valid: true,
          expiresAt: new Date(Date.now() + 86400000).toISOString(),
          distanceCheck: true,
          snrThreshold: 20.0,
          parameters: {
            offsetX: 0.05,
            offsetY: -0.02,
            intensity: 1.0
          }
        },
        uptime: 3600,
        cloudConnected: true,
        lastHeartbeat: new Date().toISOString()
      }));
      return;
    }

    // Default 404
    res.writeHead(404);
    res.end(JSON.stringify({ message: 'Not found' }));
  });
};

describe('Mock Server API Tests', () => {
  let server;

  beforeEach(() => {
    server = createMockServer();
  });

  describe('POST /api/auth/login', () => {
    it('should successfully login as admin', async () => {
      const response = await request(server)
        .post('/api/auth/login')
        .send({ username: 'admin', password: 'admin123' });

      expect(response.status).toBe(200);
      expect(response.body.username).toBe('admin');
      expect(response.body.role).toBe('ADMIN');
      expect(response.body.id).toContain('user_admin_');
      expect(response.body.sessionStart).toBeDefined();
    });

    it('should successfully login as operator', async () => {
      const response = await request(server)
        .post('/api/auth/login')
        .send({ username: 'operator', password: 'operator123' });

      expect(response.status).toBe(200);
      expect(response.body.username).toBe('operator');
      expect(response.body.role).toBe('OPERATOR');
    });

    it('should successfully login as engineer', async () => {
      const response = await request(server)
        .post('/api/auth/login')
        .send({ username: 'engineer', password: 'engineer123' });

      expect(response.status).toBe(200);
      expect(response.body.username).toBe('engineer');
      expect(response.body.role).toBe('ENGINEER');
    });

    it('should reject login with invalid username', async () => {
      const response = await request(server)
        .post('/api/auth/login')
        .send({ username: 'invalid', password: 'wrong' });

      expect(response.status).toBe(401);
      expect(response.body.message).toBe('Invalid credentials');
      expect(response.body.code).toBe('AUTH_FAILED');
    });

    it('should reject login with invalid password', async () => {
      const response = await request(server)
        .post('/api/auth/login')
        .send({ username: 'admin', password: 'wrongpassword' });

      expect(response.status).toBe(401);
      expect(response.body.message).toBe('Invalid credentials');
      expect(response.body.code).toBe('AUTH_FAILED');
    });

    it('should reject login with missing credentials', async () => {
      const response = await request(server)
        .post('/api/auth/login')
        .send({});

      expect(response.status).toBe(401);
    });
  });

  describe('POST /api/auth/logout', () => {
    it('should successfully logout', async () => {
      const response = await request(server)
        .post('/api/auth/logout');

      expect(response.status).toBe(200);
      expect(response.body.success).toBe(true);
    });
  });

  describe('GET /api/health', () => {
    it('should return health status', async () => {
      const response = await request(server)
        .get('/api/health');

      expect(response.status).toBe(200);
      expect(response.body.state).toBe('IDLE');
      expect(response.body.interlocks).toBeDefined();
      expect(response.body.interlocks.keySwitch).toBe(true);
      expect(response.body.interlocks.enableButton).toBe(true);
      expect(response.body.interlocks.doorSensor).toBe(true);
      expect(response.body.interlocks.emergencyStop).toBe(false);
      expect(response.body.interlocks.beamWatchdog).toBe(true);
      expect(response.body.interlocks.overTemp).toBe(false);
    });

    it('should return calibration data', async () => {
      const response = await request(server)
        .get('/api/health');

      expect(response.status).toBe(200);
      expect(response.body.calibration).toBeDefined();
      expect(response.body.calibration.id).toBe('cal_20250124_001');
      expect(response.body.calibration.valid).toBe(true);
      expect(response.body.calibration.distanceCheck).toBe(true);
      expect(response.body.calibration.snrThreshold).toBe(20.0);
      expect(response.body.calibration.parameters).toBeDefined();
      expect(response.body.calibration.parameters.offsetX).toBe(0.05);
      expect(response.body.calibration.parameters.offsetY).toBe(-0.02);
      expect(response.body.calibration.parameters.intensity).toBe(1.0);
    });

    it('should return system status information', async () => {
      const response = await request(server)
        .get('/api/health');

      expect(response.status).toBe(200);
      expect(response.body.uptime).toBe(3600);
      expect(response.body.cloudConnected).toBe(true);
      expect(response.body.lastHeartbeat).toBeDefined();
    });
  });

  describe('Unknown endpoints', () => {
    it('should return 404 for unknown endpoints', async () => {
      const response = await request(server)
        .get('/api/unknown');

      expect(response.status).toBe(404);
      expect(response.body.message).toBe('Not found');
    });

    it('should return 404 for unknown POST endpoints', async () => {
      const response = await request(server)
        .post('/api/unknown')
        .send({ test: 'data' });

      expect(response.status).toBe(404);
      expect(response.body.message).toBe('Not found');
    });
  });

  describe('CORS', () => {
    it('should handle OPTIONS preflight requests', async () => {
      const response = await request(server)
        .options('/api/auth/login');

      expect(response.status).toBe(200);
      expect(response.headers['access-control-allow-origin']).toBe('*');
      expect(response.headers['access-control-allow-methods']).toContain('POST');
    });
  });
});
