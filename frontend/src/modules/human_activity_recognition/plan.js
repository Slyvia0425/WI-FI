// Plan for this module. Statuses describe implemented code versus work still requiring the real dataset.
const p = (t, s = 'planned') => ({ t, s })

export default {
  preprocessing: [
    p("Read HumanActivityRecognition HDF5, metadata and official splits; keep user, environment, device ids", 'done'),
    p("Validate labels, shapes, finite values, signal quality and split-group leakage", 'done'),
    p("Fit normalisation on training data only and serialise feature statistics with the model", 'done'),
  ],
  training: [
    p("1D-CNN baseline for local temporal features", 'done'),
    p("CNN-Transformer for longer-range changes", 'done'),
    p("Deterministic seeds, class weights, early stopping and checkpoint metadata", 'done'),
  ],
  experiments: [
    p("Compare 1D-CNN and CNN-Transformer on identical splits and seeds", 'doing'),
    p("Run three seeds and report mean and standard deviation with the real dataset", 'doing'),
  ],
  evaluation: [
    p("Macro-F1, per-class precision and recall, confusion matrix and calibration error", 'done'),
    p("OOD results across user, environment and device with transition and quality slices", 'doing'),
    p("Export activity, confidence and stable fields plus probabilities and model version", 'done'),
  ],
}
