// Constants of the page that are not data: timing of the replays, the model shown first, and the plain-language names of the ten statistics.
// Numbers about the data and the models (shapes, counts, settings) are not here: they come from data/replay.json, which the backend writes from config.json.

export const PLAY_INTERVAL_MS = 800                  // time between two steps of a replay (preview windows, trees, iterations)
export const DEFAULT_TRAINING_MODEL = 'logistic_regression'   // the model whose training is shown first (the chosen one)

export const STAT_TEXT = [['mean', 'average level'], ['std', 'how much it moves'], ['min', 'lowest value'], ['max', 'highest value'], ['skew', 'leans to one side?'], ['kurtosis', 'spiky or flat?'],
  ['mean_abs_diff', 'average change per step'], ['dominant_freq', 'strongest rhythm (Hz)'], ['low_band_energy', 'slow motion, 0.5–3 Hz'], ['high_band_energy', 'fast motion, 3–20 Hz']]
