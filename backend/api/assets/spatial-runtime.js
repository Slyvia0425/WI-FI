(() => {
  const lite = location.pathname.startsWith('/spatial-lite/')
  if (!lite) return
  function apply() {
    const source = document.getElementById('source')
    if (!source) return
    for (const option of [...source.options]) {
      const supported = ['live', '1', '2', '3'].includes(option.value)
      if (!supported) { option.remove(); continue }
      if (['1', '2', '3'].includes(option.value)) {
        const text = `Examples: ${option.value} ${option.value === '1' ? 'object' : 'objects'}`
        option.dataset.en = text; option.textContent = text
      }
    }
    const summary = document.querySelector('#results-panel > summary > span')
    if (summary) { summary.dataset.en = 'Saved evaluation'; summary.textContent = 'Saved evaluation' }
  }
  const source = document.getElementById('source')
  if (source) new MutationObserver(apply).observe(source, { childList: true })
  new MutationObserver(() => {
    const summary = document.querySelector('#results-panel > summary > span')
    if (summary && summary.textContent !== 'Saved evaluation') apply()
  }).observe(document.getElementById('results-panel'), { childList: true, subtree: true })
  apply()
})()
