// Current integration state; original training source is preserved separately.
const p = (t, s = 'done') => ({ t, s })
export default {
  preprocessing: [
    p('Read exactly the same HP raw amplitude [232,500] as C; match raw-input identity'),
    p('Use the released spectral branch: 4195 features, fixed selector keeps 2800'),
    p('Keep B spectral preprocessing separate from C ResNet per-record normalization'),
  ],
  training: [
    p('Load author-released ExtraTrees seed42, 900 trees; no new training'),
    p("Preserve the author's CNN and other-seed artifacts separately"),
  ],
  experiments: [
    p('Run frozen B on C HP validation inputs as advisory context'),
    p('B has five activity classes and no Fall class; no score fusion or veto'),
  ],
  evaluation: [
    p('Verify paired raw-input identity and preserve C scores and threshold'),
    p('Export activity, five-class probabilities, confidence flag and model version'),
    p('Evaluate joint performance with activity ground truth and audited cross-task splits', 'planned'),
  ],
}
