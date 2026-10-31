export function Notice({ kind = 'info', children }) {
  return <div className={`notice ${kind}`}>{children}</div>
}

export const DISCLAIMER =
  'Research prototype · Recorded results and signal simulation · Modules use independent data; no joint accuracy is claimed.'
