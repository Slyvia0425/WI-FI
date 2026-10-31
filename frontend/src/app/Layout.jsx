import { useEffect } from 'react'
import { NavLink, Link, Outlet, useLocation } from 'react-router-dom'
import { MODULES } from '../modules/registry.js'
import { DISCLAIMER } from '../components/ui.jsx'

export default function Layout() {
  const { pathname } = useLocation()
  useEffect(() => { window.scrollTo({ top: 0, behavior: 'instant' }) }, [pathname])
  return (
    <div className="page">
      <div className="wrap">
        <header className="top">
          <Link to="/" className="brand" aria-label="WiFi Sense home">
            <svg width="32" height="32" viewBox="0 0 32 32" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" aria-hidden="true">
              <path d="M4 12a17 17 0 0 1 24 0" /><path d="M8 17a11 11 0 0 1 16 0" /><path d="M12 22a5 5 0 0 1 8 0" />
              <circle cx="16" cy="26" r="1.4" fill="currentColor" />
            </svg>
            <span>WiFi Sense</span>
          </Link>
          <nav className="topnav">
            <NavLink to="/" end>Home</NavLink>
            {MODULES.map((m) => (
              <NavLink key={m.id} to={`/module/${m.id}`}>{m.navLabel}</NavLink>
            ))}
            <NavLink to="/fusion">Fusion</NavLink>
          </nav>
          <div className="badge"><i />GROUP 66</div>
        </header>
        <Outlet />
        <footer className="foot">{DISCLAIMER}</footer>
      </div>
    </div>
  )
}
