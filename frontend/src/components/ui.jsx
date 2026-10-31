export function Notice({ kind = 'info', children }) {
  return <div className={`notice ${kind}`}>{children}</div>
}

export const DISCLAIMER =
  'Research prototype for single-person indoor monitoring. Not a medical device. B and C share HP input segments; other modules use independent data. The five-module room-state view is illustrative, without joint ground truth.'
