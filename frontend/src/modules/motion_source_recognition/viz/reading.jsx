import { num, pct } from './theme.js'

// The short reading beside each chart of the evaluation stage: what the chart shows, why, and what could help. Every number is taken from the chart's own data.

const PARTS = [['result', 'WHAT WE SEE'], ['why', 'WHY'], ['next', 'WHAT COULD HELP']]

// **text** in a bullet is highlighted.
export const Bullet = ({ text }) => <li>{text.split('**').map((t, i) => (i % 2 ? <b key={i}>{t}</b> : t))}</li>

export function Reading({ reading }) {
  return (
    <aside className="card msr-reading" aria-label="How to read this chart">
      {PARTS.map(([key, label]) => (
        <section key={key}>
          <div className="msr-reading-kicker">{label}</div>
          <ul>{reading[key].map((t, i) => <Bullet key={i} text={t} />)}</ul>
        </section>
      ))}
    </aside>
  )
}

const byClass = (rows, c) => rows.find((r) => r.class === c)
const windows = (table, c) => Number(String(table.find((r) => r.Class === c).Windows).replace(/,/g, ''))

export function readingFor(tab, dash, data, idx) {
  const table = data.preprocessing.table
  const total = table.reduce((s, r) => s + Number(String(r.Windows).replace(/,/g, '')), 0)
  const individuals = (c) => table.find((r) => r.Class === c).Individuals
  if (tab === 'scores') {
    const values = dash.models.map((m) => m.test.macro_f1[0])
    const rare = (windows(table, 'IRobot') + windows(table, 'Fan')) / total
    return {
      result: [`All five models: macro-F1 **${num(Math.min(...values))} to ${num(Math.max(...values))}**, within the spread`, 'Human and Pet are good', 'IRobot and Fan are not'],
      why: [`IRobot and Fan are only **${pct(rare)}** of the windows`, `They come from **${individuals('IRobot')} and ${individuals('Fan')}** individuals`, 'A model learns those devices, not the class'],
      next: ['More robot and fan recordings, in more rooms', 'Other features, models and averaging (Stage 3) changed little'],
    }
  }
  if (tab === 'classes') {
    const o = dash.outcome
    const [H, P, R, F] = ['Human', 'Pet', 'IRobot', 'Fan'].map((c) => byClass(o, c))
    return {
      result: [`Human **${pct(H.without_rule.right)}** and Pet **${pct(P.without_rule.right)}** are mostly right`, `IRobot **${pct(R.without_rule.right)}** and Fan **${pct(F.without_rule.right)}** almost never`],
      why: [`IRobot: **${individuals('IRobot')}** individuals. Fan: **${individuals('Fan')}**`, 'The test devices were never seen in training'],
      next: [`Unknown rule: **${pct(R.with_rule.unknown)}** of IRobot and **${pct(F.with_rule.unknown)}** of Fan become unknown, not wrong`, 'The real fix is more devices'],
    }
  }
  if (tab === 'confusion') {
    const { labels, matrix } = dash.confusion
    const share = (i, j) => matrix[i][j] / matrix[i].reduce((s, v) => s + v, 0)
    const worst = (i) => labels.map((_, j) => j).filter((j) => j !== i).sort((a, b) => share(i, b) - share(i, a))[0]
    const [r, f] = [labels.indexOf('IRobot'), labels.indexOf('Fan')]
    return {
      result: [`IRobot answered as ${labels[worst(r)]}: **${pct(share(r, worst(r)))}**`, `Fan answered as ${labels[worst(f)]}: **${pct(share(f, worst(f)))}**`, 'Human and Pet sit on the diagonal'],
      why: ['An unseen robot or fan looks closest to the two classes the model knows best'],
      next: ['More data from other devices', 'A threshold per class was tried and was unstable'],
    }
  }
  if (tab === 'rooms') {
    const [seen, , fresh] = dash.compare
    const sorted = [...dash.rooms].sort((a, b) => b.accuracy - a.accuracy)
    return {
      result: [`Seen rooms: **${num(seen.value)}** accuracy`, `New room: **${num(fresh.value)}**, like guessing`, `By room: ${num(sorted[sorted.length - 1].accuracy)} to ${num(sorted[0].accuracy)}`],
      why: ['Each person, pet and device was recorded in one room', 'The model also learns the room and the individual', 'A new room changes both at once'],
      next: ['Record the same individuals in several rooms', 'Other features did not close the gap (Stage 3)'],
    }
  }
  if (tab === 'unknown') {
    const cur = dash.tradeoff[idx], base = dash.tradeoff[0]
    return {
      result: cur.share === 0
        ? [`Every window is answered: accuracy **${num(cur.accuracy_kept)}**`, 'Move the slider to answer unknown more often']
        : [`**${pct(cur.share)}** unknown: accuracy of the answered ${num(base.accuracy_kept)} → **${num(cur.accuracy_kept)}**`, `**${pct(cur.errors_rejected)}** of the mistakes held back`],
      why: ['Wrong answers tend to have lower confidence', 'But many are confident, so some get through'],
      next: ['Better calibration or more data would catch more', 'On unseen rooms and devices, even more windows become unknown'],
    }
  }
  if (tab === 'speed') {
    const windowMs = (dash.feature_demo.steps / dash.feature_demo.sampling_rate_hz) * 1000
    const main = dash.speed.find((s) => /Logistic Regression on MiniRocket/.test(s.name)) || dash.speed[0]
    return {
      result: [`**${Math.round(main.ms)} ms** per window`, `**${pct(main.ms / windowMs, 1)}** of the ${windowMs.toLocaleString()} ms between two windows`],
      why: [`A linear model on MiniRocket is cheap: **${main.megabytes} MB**`, `Forests are larger, up to ${Math.max(...dash.speed.map((s) => s.megabytes))} MB`],
      next: ['Nothing to optimise for speed'],
    }
  }
  return null
}
