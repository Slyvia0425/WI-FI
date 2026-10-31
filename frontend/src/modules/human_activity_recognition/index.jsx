import ModulePage from '../../components/ModulePage.jsx'
import { ProbBars } from '../../components/charts.jsx'
import ActivityTimelineSection from './components/ActivityTimelineSection.jsx'
import { Notice } from '../../components/ui.jsx'

export default function HumanActivityRecognition() {
  return (
    <ModulePage
      id="human_activity_recognition"
      summary="Author-released spectral ExtraTrees seed42, run on the same raw HP segments as Module C. Five-class activity preferences provide context for independent ResNet18 fall scoring."
      stages={{ evaluation: <><Notice>Real frozen B model inference. FallDetection segments have no five-class activity annotations, so B accuracy and joint B/C accuracy are not measured. B has no Fall class and cannot veto C.</Notice><ActivityTimelineSection /></> }}
      renderFrame={(f) => {
        const confidence = Math.round((f.confidence || 0) * 100)
        const probabilities = f.probabilities || {}
        return (
          <div>
            <div className="big">
              {f.activity} <span className="sm">confidence {confidence}%</span>
            </div>
            <p className="sm">
              {f.stable ? 'B confidence cutoff passed' : f.transition ? 'Transition or unstable evidence' : 'Weak B evidence'}
              {f.quality ? ` · quality ${f.quality}` : ''}
            </p>
            <p className="sm">{f.sample_id} · confidence flag is not temporal stability or a verified activity label.</p>
            {Object.keys(probabilities).length > 0 && <ProbBars probs={probabilities} highlight={f.activity} />}
          </div>
        )
      }}
    />
  )
}
