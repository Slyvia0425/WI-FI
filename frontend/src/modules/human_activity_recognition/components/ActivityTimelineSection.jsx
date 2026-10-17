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
  return <ActivityTimeline frames={data?.frames || []} />
}
