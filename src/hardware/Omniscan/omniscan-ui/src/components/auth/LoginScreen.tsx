import { useState, useEffect } from 'react';
import { Button } from '@/components/common/Button';
import { api } from '@/services/api';
import { useStore } from '@/store/useStore';

export function LoginScreen() {
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState('');
  const [connectionStatus, setConnectionStatus] = useState<'checking' | 'connected' | 'disconnected'>('checking');
  
  const { setUser, addNotification } = useStore();

  // Check connection to orchestrator on mount
  useEffect(() => {
    const checkConnection = async () => {
      try {
        const response = await fetch('/api/connection/status');
        const data = await response.json();
        // Handle both formats: Node.js proxy {success, data: {connected}} and orchestrator {connected}
        const isConnected = data.data?.connected ?? data.connected ?? false;
        setConnectionStatus(isConnected ? 'connected' : 'disconnected');
      } catch {
        setConnectionStatus('disconnected');
      }
    };
    
    checkConnection();
    const interval = setInterval(checkConnection, 3000);
    return () => clearInterval(interval);
  }, []);

  const handleLogin = async (e: React.FormEvent) => {
    e.preventDefault();
    setError('');
    setIsLoading(true);

    try {
      console.log('[Login] Attempting login for:', username);
      const result = await api.login(username, password);
      console.log('[Login] Login result:', result);
      
      if (result.success && result.data) {
        console.log('[Login] Setting user:', result.data);
        setUser(result.data);
        addNotification(`Welcome, ${result.data.username}`, 'success');
      } else {
        console.error('[Login] Login failed:', result.error);
        setError(result.error || 'Login failed');
      }
    } catch (err) {
      console.error('[Login] Network error:', err);
      setError('Network error. Please check your connection.');
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-gray-100 flex items-center justify-center p-4">
      <div className="bg-white rounded-lg shadow-lg p-6 w-full max-w-sm">
        <div className="text-center mb-8">
          <img src="/EosDxLogo.png" alt="EosDx" className="h-16 mx-auto mb-4" />
          <h1 className="text-3xl font-bold text-gray-900 mb-2">Omniscan</h1>
          <p className="text-gray-600">Medical XRD Diagnostic System</p>
          
          {/* Connection Status */}
          <div className="mt-4">
            {connectionStatus === 'checking' && (
              <div className="flex items-center justify-center gap-2 text-sm text-gray-500">
                <div className="w-2 h-2 bg-gray-400 rounded-full animate-pulse"></div>
                Checking orchestrator connection...
              </div>
            )}
            {connectionStatus === 'connected' && (
              <div className="flex items-center justify-center gap-2 text-sm text-green-600">
                <div className="w-2 h-2 bg-green-500 rounded-full"></div>
                Orchestrator connected
              </div>
            )}
            {connectionStatus === 'disconnected' && (
              <div className="flex items-center justify-center gap-2 text-sm text-red-600">
                <div className="w-2 h-2 bg-red-500 rounded-full"></div>
                Orchestrator offline
              </div>
            )}
          </div>
        </div>

        <form onSubmit={handleLogin} className="space-y-4">
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">
              Username
            </label>
            <input
              type="text"
              value={username}
              onChange={(e) => setUsername(e.target.value)}
              required
              className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
              placeholder="Enter your username"
            />
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">
              Password
            </label>
            <input
              type="password"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              required
              className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
              placeholder="Enter your password"
            />
          </div>

          {error && (
            <div className="bg-red-50 border border-red-200 rounded p-3">
              <p className="text-sm text-red-800">{error}</p>
            </div>
          )}

          <Button
            type="submit"
            variant="primary"
            size="lg"
            disabled={isLoading || !username || !password}
            className="w-full"
          >
            {isLoading ? 'Signing in...' : 'Sign In'}
          </Button>
        </form>

        <div className="mt-6 text-center">
          <p className="text-xs text-gray-500">
            FDA-regulated medical device system<br />
            Authorized personnel only
          </p>
          <p className="text-xs text-gray-400 mt-4">
            Dev credentials: sad / good!
          </p>
        </div>
      </div>
    </div>
  );
}
