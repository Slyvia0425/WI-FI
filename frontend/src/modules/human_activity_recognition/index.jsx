import ModulePage from '../../components/ModulePage.jsx'
import { ProbBars } from '../../components/charts.jsx'
import ActivityTimelineSection from './components/ActivityTimelineSection.jsx'

export default function HumanActivityRecognition() {
  return (
    <ModulePage
      id="human_activity_recognition"
      summary="Which stable single-person state is present, how confident is the model, and when should the system report a transition or unknown?"
      stages={{ evaluation: <ActivityTimelineSection /> }}
      renderFrame={(f) => {
        const confidence = Math.round((f.confidence || 0) * 100)
        const probabilities = f.probabilities || {}
        return (
          <div>
            <div className="big">
              {f.activity} <span className="sm">confidence {confidence}%</span>
            </div>
            <p className="sm">
              {f.stable ? 'Stable state' : f.transition ? 'Transition or unstable evidence' : 'Unknown or weak evidence'}
              {f.quality ? ` · quality ${f.quality}` : ''}
            </p>
            {Object.keys(probabilities).length > 0 && <ProbBars probs={probabilities} highlight={f.activity} />}
          </div>
        )
      }}
    />
  )
}
