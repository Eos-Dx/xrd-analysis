import { createServer } from 'http';

const users = {
  'admin': { password: 'admin123', username: 'admin', role: 'admin' },
  'operator': { password: 'operator123', username: 'operator', role: 'operator' },
  'engineer': { password: 'engineer123', username: 'engineer', role: 'engineer' }
};

const server = createServer((req, res) => {
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

server.listen(8080, () => {
  console.log('Mock API server running on http://localhost:8080');
  console.log('\nTest credentials:');
  console.log('  admin / admin123');
  console.log('  operator / operator123');
  console.log('  engineer / engineer123');
});
