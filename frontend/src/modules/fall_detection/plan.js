const p = (t, s = 'planned') => ({ t, s })
export default {
  preprocessing: [
    p('CSI-Bench HP [232,500,1] amplitudes; preserve the official ResNet18 preprocessing', 'done'),
    p('Interface carries same-input identity; each module keeps its own preprocessing', 'done'),
    p('Do not match unrelated predictions by replay display time', 'done'),
  ],
  training: [
    p('Reuse the CSI-Bench official ResNet18 seed42 weights; best epoch12', 'done'),
    p('Preserve earlier 1D-CNN and TCS models as separate experiments', 'done'),
    p('No ResNet18-to-TCS fine-tuning has been performed', 'done'),
  ],
  experiments: [
    p('Run C only for paired B confidence <0.6; otherwise return not evaluated', 'done'),
    p('Accept B context only for an exact input or an explicit same-clock overlapping window', 'done'),
    p('Load author-released B seed42 and produce same-raw-input HP predictions', 'done'),
  ],
  evaluation: [
    p('Show all926 HP validation segment scores and original operating point', 'done'),
    p('New-process ResNet18 reload and raw HDF5 inference', 'done'),
    p('Report routing coverage and real Fall segments skipped by the gate', 'done'),
    p('Joint B/C accuracy on paired real recordings'),
  ],
}
