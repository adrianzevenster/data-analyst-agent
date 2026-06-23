import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      // File uploads: separate rule first so http-proxy never buffers the body
      '/api/uploads': {
        target: 'http://localhost:8099',
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/api\/uploads/, '/uploads'),
      },
      '/api': {
        target: 'http://localhost:8099',
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/api/, ''),
        configure: (proxy) => {
          proxy.on('proxyRes', (_proxyRes, req) => {
            // Ensure SSE connections are not buffered or prematurely closed.
            if (req.url?.includes('/stream')) {
              _proxyRes.headers['cache-control'] = 'no-cache'
              _proxyRes.headers['x-accel-buffering'] = 'no'
            }
          })
        },
      },
    },
  },
})
