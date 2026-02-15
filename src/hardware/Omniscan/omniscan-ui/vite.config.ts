import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import path from 'path'

export default defineConfig({
  plugins: [react()],
  resolve: {
    alias: {
      '@': path.resolve(__dirname, './src'),
    },
  },
  server: {
    port: 3000,
    proxy: {
      // Route REST traffic through the local proxy server (server.js)
      '/api': {
        target: 'http://localhost:3001',  // UI proxy -> orchestrator
        changeOrigin: true,
      },
      // WebSocket can continue to go directly to orchestrator
      '/ws': {
        target: 'ws://localhost:8080',  // Orchestrator WebSocket
        ws: true,
      },
    },
  },
})
