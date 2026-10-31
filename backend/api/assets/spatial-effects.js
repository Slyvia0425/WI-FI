(() => {
  const NS = 'http://www.w3.org/2000/svg'
  const reduced = matchMedia('(prefers-reduced-motion: reduce)')
  let enabled = !reduced.matches, visible = true, raf = 0, last = 0, width = 0, height = 0
  const canvas = document.createElement('canvas')
  canvas.id = 'space-particles'
  canvas.setAttribute('aria-hidden', 'true')
  document.body.prepend(canvas)
  const ctx = canvas.getContext('2d')
  const dots = Array.from({ length: 34 }, (_, i) => ({ x: Math.random(), y: Math.random(), speed: .007 + Math.random() * .011, r: i % 4 ? 1 : 1.7 }))
  const resize = () => {
    width = innerWidth; height = innerHeight
    const ratio = Math.min(devicePixelRatio || 1, 1.5)
    canvas.width = width * ratio; canvas.height = height * ratio
    ctx?.setTransform(ratio, 0, 0, ratio, 0, 0)
  }
  function draw(now) {
    raf = 0
    if (!enabled || !visible || document.hidden || reduced.matches || !ctx) return
    raf = requestAnimationFrame(draw)
    if (now - last < 33) return
    const dt = Math.min((now - (last || now)) / 1000, .06); last = now
    ctx.clearRect(0, 0, width, height)
    for (const p of dots) {
      p.y -= p.speed * dt; p.x += Math.sin(now * .00013 + p.y * 9) * dt * .0015
      if (p.y < -.02) { p.y = 1.02; p.x = Math.random() }
      ctx.beginPath(); ctx.arc(p.x * width, p.y * height, p.r, 0, Math.PI * 2)
      ctx.fillStyle = p.r > 1 ? '#9b8dfc7a' : '#65e4e263'; ctx.fill()
    }
  }
  function sync() {
    document.body.classList.toggle('fx-off', !enabled || reduced.matches)
    document.body.classList.toggle('fx-suspended', !visible || document.hidden)
    if (raf) cancelAnimationFrame(raf)
    raf = 0; last = 0
    if (enabled && visible && !document.hidden && !reduced.matches) raf = requestAnimationFrame(draw)
    else ctx?.clearRect(0, 0, width, height)
  }
  addEventListener('resize', resize, { passive: true })
  addEventListener('message', e => {
    if (e.origin !== location.origin || e.source !== parent || e.data?.type !== 'wifi-space-effects') return
    enabled = e.data.enabled === true; visible = e.data.visible !== false; sync()
  })
  document.addEventListener('visibilitychange', sync)
  reduced.addEventListener('change', sync)
  document.addEventListener('pointermove', e => {
    if (!enabled || reduced.matches || e.pointerType === 'touch') return
    const panel = e.target.closest('.maps article,.workspace')
    if (!panel) return
    const rect = panel.getBoundingClientRect()
    panel.style.setProperty('--fx-x', `${e.clientX - rect.left}px`)
    panel.style.setProperty('--fx-y', `${e.clientY - rect.top}px`)
  }, { passive: true })
  document.addEventListener('click', e => {
    if (!enabled || reduced.matches || !e.target.closest('button:not(:disabled),summary')) return
    const ripple = document.createElement('i')
    ripple.className = 'fx-ripple'; ripple.setAttribute('aria-hidden', 'true')
    const rect = e.target.getBoundingClientRect()
    ripple.style.left = `${e.detail ? e.clientX : rect.left + rect.width / 2}px`
    ripple.style.top = `${e.detail ? e.clientY : rect.top + rect.height / 2}px`
    document.body.append(ripple)
    setTimeout(() => ripple.remove(), 700)
  }, { passive: true })
  function node(tag, attributes) {
    const el = document.createElementNS(NS, tag)
    for (const [key, value] of Object.entries(attributes)) el.setAttribute(key, value)
    return el
  }
  function decorateMap() {
    const svg = document.querySelector('#truth-map svg')
    if (!svg || svg.querySelector('.fx-signal-layer')) return
    const tx = [...svg.querySelectorAll('circle[fill="#68bbff"]')]
    const rx = [...svg.querySelectorAll('circle[fill="#ffbf73"]')]
    if (!tx.length || !rx.length) return
    const layer = node('g', { class: 'fx-signal-layer', 'aria-hidden': 'true' })
    tx.forEach((t, i) => {
      const x = t.getAttribute('cx'), y = t.getAttribute('cy')
      const ring = node('circle', { class: 'fx-ring', cx: x, cy: y, r: 5 })
      ring.style.animationDelay = `${-i * 1.2}s`; layer.append(ring)
      for (const offset of [0, 3]) {
        const r = rx[(i * 2 + offset) % rx.length]
        const d = `M${x},${y} L${r.getAttribute('cx')},${r.getAttribute('cy')}`
        layer.append(node('path', { d, class: 'fx-link' }))
        const flow = node('path', { d, class: 'fx-flow', pathLength: 100 })
        flow.style.animationDelay = `${-(i + offset) * .7}s`; layer.append(flow)
      }
    })
    svg.insertBefore(layer, svg.children[1] || null)
  }
  function stateChanged() {
    const busy = document.querySelector('#status')?.dataset.tone === 'running'
    document.body.classList.toggle('fx-running', busy)
  }
  const truth = document.getElementById('truth-map')
  const status = document.getElementById('status')
  const decision = document.getElementById('decision')
  if (truth) new MutationObserver(decorateMap).observe(truth, { childList: true })
  if (status) new MutationObserver(stateChanged).observe(status, { childList: true, attributes: true, attributeFilter: ['data-tone'] })
  let flashTimer
  if (decision) new MutationObserver(() => {
    if (!enabled || reduced.matches) return
    document.body.classList.remove('fx-result')
    clearTimeout(flashTimer)
    requestAnimationFrame(() => {
      document.body.classList.add('fx-result')
      flashTimer = setTimeout(() => document.body.classList.remove('fx-result'), 850)
    })
  }).observe(decision, { childList: true })
  resize(); decorateMap(); stateChanged(); sync()
  addEventListener('pagehide', () => { if (raf) cancelAnimationFrame(raf); clearTimeout(flashTimer) }, { once: true })
})()
