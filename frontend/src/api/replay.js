import { useEffect, useState } from 'react'

// API client: how the website gets each module's replay JSON.
// Reads a module's replay JSON.
// With VITE_BACKEND_URL set, it asks the backend (GET <url>/api/<module>/replay) and falls back to the
// bundled src/modules/<module>/data/replay.json if the backend is unreachable or has no output yet.
// Without it, the bundled file is used.
export async function fetchModule(mod) {
  const backend = import.meta.env.VITE_BACKEND_URL ?? ''
  let json = null
  let source = 'bundled'
  {
    try {
      const res = await fetch(`${backend.replace(/\/$/, '')}/api/${mod.id}/replay`)
      if (res.ok) { json = await res.json(); source = 'backend' }
    } catch { /* fall back below */ }
  }
  if (!json) json = mod.bundled
  if (!json) throw new Error(`${mod.title}: no data (add src/modules/${mod.id}/data/replay.json or export from the backend)`)
  if (!Array.isArray(json.frames)) throw new Error(`${mod.title}: data has no "frames" array`)
  return { ...json, _source: source }
}

// { data, error, loading } for one module's replay JSON.
export function useModuleData(mod) {
  const [state, setState] = useState({ data: null, error: null, loading: true })
  useEffect(() => {
    let live = true
    fetchModule(mod)
      .then((data) => live && setState({ data, error: null, loading: false }))
      .catch((error) => live && setState({ data: null, error, loading: false }))
    return () => {
      live = false
    }
  }, [mod.id])
  return state
}

// Last frame whose t <= time, or null if the module has not reported yet.
export function frameAt(data, time) {
  if (!data) return null
  let hit = null
  for (const f of data.frames) if (f.t <= time) hit = f
  return hit
}

// Current frame of every module, keyed by role: { motionSource, activity, fall, breathing, space }.
export function framesByRole(modules, all, time) {
  const out = {}
  modules.forEach((m) => { out[m.role] = frameAt(all[m.id], time) })
  return out
}

// Loads every module once. Returns { all: { <module id>: json | null }, ready }.
export function useAllModules(modules) {
  const [state, setState] = useState({ all: {}, ready: false })
  useEffect(() => {
    let live = true
    Promise.allSettled(modules.map((m) => fetchModule(m))).then((res) => {
      if (!live) return
      const all = {}
      res.forEach((r, i) => { all[modules[i].id] = r.status === 'fulfilled' ? r.value : null })
      setState({ all, ready: true })
    })
    return () => { live = false }
  }, [])
  return state
}
