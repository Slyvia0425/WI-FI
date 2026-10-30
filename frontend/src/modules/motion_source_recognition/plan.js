// Plan for this module. Edit freely: add, remove or reword items, and move `s` from 'planned' to 'doing' to 'done'.
// The items describe what was done; the results are in data/replay.json (written by backend evaluation/export_replay.py).
const p = (t, s = 'planned') => ({ t, s })

export default {
  preprocessing: [
    p("Read the HDF5 windows and metadata; quality checks on every window (60,858 windows, none flagged)", 'done'),
    p("Split by whole recording groups into 3 train, validation and test splits; identical windows never cross sets", 'done'),
    p("Features: time-series statistics (2,320), MiniRocket (9,996) and 60 physical numbers", 'done'),
  ],
  training: [
    p("Random Forest on time-series statistics as the baseline", 'done'),
    p("Logistic Regression on MiniRocket features for comparison", 'done'),
    p("Random Forest on MiniRocket, and both classifiers on statistics + physical features, as further comparisons", 'done'),
  ],
  experiments: [
    p("Compare features against algorithms, averaging, smoothing, calibration, gradient boosting, spectral and physical features", 'done'),
    p("Choose the final model on the validation rows before the test rows are scored; check new rooms, speed and size", 'done'),
    p("Repeat over 3 grouped splits and report the mean and standard deviation", 'done'),
  ],
  evaluation: [
    p("Test scores of every model and of two reference lines; Fan and IRobot are weak and shown per class", 'done'),
    p("Confusion matrix and error analysis: Human vs Fan, Pet vs Human, errors by room and recording", 'done'),
    p("Unknown rule from a calibrated confidence threshold; the output fields for the other modules", 'done'),
  ],
}
