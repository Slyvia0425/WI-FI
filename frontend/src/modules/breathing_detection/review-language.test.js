import test from 'node:test'
import assert from 'node:assert/strict'
import { createElement } from 'react'
import { renderToStaticMarkup } from 'react-dom/server'
import { localizeTree, reviewText } from './review-language.js'
import { eventsAt } from './session.js'
import { readFileSync } from 'node:fs'

test('English remains the default and does not transform the original tree', () => {
  const node = createElement('p', null, 'Breathing workspace')
  assert.equal(localizeTree(node), node)
  assert.equal(reviewText('Breathing workspace'), 'Breathing workspace')
  assert.equal(reviewText('Breathing workspace', 'zh'), '呼吸监测工作台')
})

test('Chinese presentation preserves handlers, values, keys, IDs and code identifiers', () => {
  const click = () => {}
  const node = createElement('section', { className: 'breathing', id: 'monitor' },
    createElement('button', { key: 'start', onClick: click, 'aria-label': 'Start observation' }, 'Start observation'),
    createElement('option', { value: 'recorded' }, 'As recorded'),
    createElement('code', null, 'Subject03_Foetus_Quiet'))
  const result = localizeTree(node, 'zh')
  const [button, option, code] = result.props.children
  assert.equal(result.props.className, 'breathing')
  assert.equal(result.props.id, 'monitor')
  assert.equal(button.props.onClick, click)
  assert.equal(button.key, 'start')
  assert.equal(button.props['aria-label'], '开始观察')
  assert.equal(option.props.value, 'recorded')
  assert.equal(option.props.children, '按原始录制回放')
  assert.equal(code.props.children, 'Subject03_Foetus_Quiet')
  assert.match(renderToStaticMarkup(result), /开始观察/)
})

test('dynamic window text, metadata and experimental units are translated without changing numbers', () => {
  assert.equal(reviewText('Latest complete 14.8–24.8s window', 'zh'), '最近一个完整窗口：14.8–24.8 秒')
  assert.equal(reviewText('0.0–10.0s window · model evidence · 18.2 BPM (experimental)', 'zh'), '0.0–10.0 秒窗口 · 模型证据 · 18.2 次/分钟（实验性）')
  assert.equal(reviewText('Subject03 · Foetus · Quiet', 'zh'), '受试者03 · 蜷缩侧卧 · 安静')
  assert.equal(reviewText('18.2 BPM · experimental', 'zh'), '18.2 次/分钟 · 实验性')
})

test('risk disclosures retain their meaning in Chinese', () => {
  assert.match(reviewText('Temporal rejection and rate-reliability gates remain disabled. Insufficient evidence does not establish absence of breathing.', 'zh'), /仍未启用.*不代表没有呼吸/)
  assert.match(reviewText('Past results are not live measurements. One short recording cannot establish long-term stability.', 'zh'), /不是实时测量.*不能证明长期稳定性/)
})

test('all recorded replay event titles and details have Chinese review copy', () => {
  const data = JSON.parse(readFileSync(new URL('./data/replay.json', import.meta.url), 'utf8'))
  for (const record of data.records) for (const sample of record.samples) for (const mode of ['recorded', 'interrupted']) {
    for (const event of eventsAt(sample, 34, data.threshold, mode)) {
      assert.match(reviewText(event.title, 'zh'), /[\u4e00-\u9fff]/)
      assert.match(reviewText(event.detail, 'zh'), /[\u4e00-\u9fff]/)
      assert.equal(reviewText(event.detail), event.detail)
    }
  }
})
