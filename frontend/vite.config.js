import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// The backend is reached through VITE_BACKEND_URL (see .env.example), so no proxy is needed here.
export default defineConfig({
  plugins: [react()],
  server: { proxy: { '/api': 'http://127.0.0.1:8000', '/spatial': 'http://127.0.0.1:8000', '/spatial-lite': 'http://127.0.0.1:8000' } },
})
