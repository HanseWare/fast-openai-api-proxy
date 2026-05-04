import { fileURLToPath, URL } from 'node:url'

import { defineConfig } from 'vite'
import vue from '@vitejs/plugin-vue'

// https://vite.dev/config/
export default defineConfig({
  plugins: [vue()],
  resolve: {
    alias: {
      '@': fileURLToPath(new URL('./src', import.meta.url))
    }
  },
  server: {
    proxy: {
      // Proxying /api
      '/api': {
        target: 'http://localhost:8000', // Replace 8080 with your backend port
        changeOrigin: true,
        // rewrite: (path) => path.replace(/^\/api/, '') // Uncomment if your backend doesn't expect the /api prefix
      },
      // Proxying /v1
      '/v1': {
        target: 'http://localhost:8000',
        changeOrigin: true,
      }
    }
  }
})
