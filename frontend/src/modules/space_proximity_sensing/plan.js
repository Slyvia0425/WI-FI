const p = (t, s = 'done') => ({ t, s })
export default {
  preprocessing: [p('Generate independent static-object scenes with simulated Wi-Fi channels'), p('Extract calibrated signal features using a fixed empty-room reference')],
  training: [p('Train zone-count models for default and T1 transmitter layouts'), p('Freeze models, confidence thresholds and source bindings')],
  experiments: [p('Compare baseline and refined zone-count models'), p('Test shared zones, object sizes and transmitter layouts')],
  evaluation: [p('Report exact scene accuracy, zone metrics and confirmed-result coverage'), p('Preserve Uncertain results and known failure cases'), p('Validate against measured signals in additional rooms', 'planned')],
}
