import { useModuleData } from '../../../api/replay.js'
import bundled from '../data/replay.json'
import ActivityTimeline from './ActivityTimeline.jsx'

const module = {
  id: 'human_activity_recognition',
  title: 'Human activity recognition',
  bundled,
}

export default function ActivityTimelineSection() {
  const { data } = useModuleData(module)
  if (data?.timeline_kind === 'presentation_only') return <p className="sm">{data.frames.length} independent segments. The output preview uses sample position, not a continuous timeline. Each input is matched to C by its raw-amplitude identity.</p>
  return <ActivityTimeline frames={data?.frames || []} />
}
