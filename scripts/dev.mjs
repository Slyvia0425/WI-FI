// Starts the backend (port 8000) and the frontend (port 5173) together. Ctrl+C stops both.
// Usage: npm run dev   (from the repository root)
import { spawn } from 'node:child_process'

const python = process.env.PYTHON || (process.platform === 'win32' ? 'python' : 'python3')
const npm = process.platform === 'win32' ? 'npm.cmd' : 'npm'

const children = [
  spawn(python, ['scripts/serve_integrated.py', '--api-only', '--port', '8000'], { stdio: 'inherit' }),
  spawn(npm, ['--prefix', 'frontend', 'run', 'dev'], { stdio: 'inherit', shell: process.platform === 'win32', env: { ...process.env, VITE_BACKEND_URL: '' } }),
]

const stop = () => children.forEach((c) => c.kill())
process.on('SIGINT', stop)
process.on('SIGTERM', stop)
children.forEach((c) => c.on('exit', stop))
