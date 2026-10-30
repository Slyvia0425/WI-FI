import { useRef, useState } from 'react'
import { MetricTable } from '../../../components/charts.jsx'

// Shared pieces of the charts: a tooltip that also opens on keyboard focus, segmented controls, tabs, and the chart card with its table view.

export function useTip() {
  const ref = useRef(null)
  const [tip, setTip] = useState(null)
  const place = (x, y, content) => {
    const r = ref.current?.getBoundingClientRect()
    if (r) setTip({ x: x - r.left, y: y - r.top, content })
  }
  const bind = (content) => ({
    tabIndex: 0,
    onPointerMove: (e) => place(e.clientX, e.clientY, content),
    onPointerLeave: () => setTip(null),
    onFocus: (e) => { const b = e.currentTarget.getBoundingClientRect(); place(b.left + b.width / 2, b.top, content) },
    onBlur: () => setTip(null),
  })
  const node = tip && <div className="msr-tip" style={{ left: tip.x, top: tip.y }}>{tip.content}</div>
  return { ref, bind, node }
}

export const Tip = ({ value, label, lines = [] }) => (
  <>
    <div className="msr-tip-value">{value}</div>
    <div className="msr-tip-label">{label}</div>
    {lines.map((l, i) => <div className="msr-tip-line" key={i}>{l}</div>)}
  </>
)

export function Segmented({ options, value, onChange, label }) {
  return (
    <div className="msr-seg" role="group" aria-label={label}>
      {options.map((o) => (
        <button key={o.value} type="button" className={o.value === value ? 'on' : ''} aria-pressed={o.value === value} onClick={() => onChange(o.value)}>{o.label}</button>
      ))}
    </div>
  )
}

export function Tabs({ tabs, value, onChange }) {
  return (
    <div className="msr-tabs" role="tablist">
      {tabs.map((t) => (
        <button key={t.value} type="button" role="tab" aria-selected={t.value === value} className={t.value === value ? 'on' : ''} onClick={() => onChange(t.value)}>{t.label}</button>
      ))}
    </div>
  )
}

// A chart card: title, one short hint, controls, and a Chart / Table switch (the table is the same data as numbers).
export function ChartCard({ title, hint, controls, table, children }) {
  const [view, setView] = useState('chart')
  return (
    <section className="card msr-card">
      <div className="msr-card-head">
        <div>
          <h3>{title}</h3>
          {hint && <div className="msr-hint">{hint}</div>}
        </div>
        <div className="msr-card-tools">
          {controls}
          {table && <Segmented label="View" value={view} onChange={setView} options={[{ value: 'chart', label: 'Chart' }, { value: 'table', label: 'Table' }]} />}
        </div>
      </div>
      {view === 'table' && table ? <div className="tblwrap"><MetricTable rows={table} /></div> : children}
    </section>
  )
}

export function Legend({ items }) {
  return (
    <div className="msr-legend">
      {items.map((it) => (
        <span key={it.label} className={it.dim ? 'dim' : ''} style={it.textColor ? { color: it.textColor } : undefined}>{it.color !== 'transparent' && <i style={{ background: it.color, borderRadius: it.line ? 0 : it.ring ? 5 : 3, height: it.line ? 3 : 10, width: it.line ? 16 : 10, boxShadow: it.ring ? `inset 0 0 0 2px ${it.ring}` : undefined }} />}{it.label}</span>
      ))}
    </div>
  )
}
