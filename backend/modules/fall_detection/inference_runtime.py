"""Shared frozen-model inference; no dependency on train.py or training memory."""
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

if __package__:
    from .models.cnn import FallCNN
else:
    from models.cnn import FallCNN


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def metrics(labels, scores, threshold):
    labels = np.asarray(labels, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    if not np.isin(labels, [0, 1]).all() or not np.isfinite(scores).all():
        raise ValueError('Invalid labels or scores')
    y, p = labels.astype(bool), scores >= float(threshold)
    tp, fp = int((y & p).sum()), int((~y & p).sum())
    fn, tn = int((y & ~p).sum()), int((~y & ~p).sum())
    return {'n': len(y), 'fall': int(y.sum()), 'nonfall': int((~y).sum()),
            'precision': tp/(tp+fp) if tp+fp else 0., 'recall': tp/(tp+fn) if tp+fn else 0.,
            'f1': 2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.,
            'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn}


def load_model(checkpoint, device='cpu', require_frozen=True):
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    if device == 'mps' and not torch.backends.mps.is_available():
        raise ValueError('MPS not available')
    checkpoint = Path(checkpoint)
    state = torch.load(checkpoint, map_location='cpu', weights_only=True)
    if require_frozen:
        receipt = json.loads(checkpoint.with_suffix('.json').read_text())
        if not state.get('frozen') or sha256(checkpoint) != receipt['checkpoint_sha256']:
            raise ValueError('Frozen checkpoint integrity check failed')
        if state['selected_threshold'] != receipt['selected_threshold']:
            raise ValueError('Frozen threshold mismatch')
        if not 0 <= state['selected_threshold'] <= 1:
            raise ValueError('Invalid frozen threshold')
    model = FallCNN(**state['model_config']).to(device)
    model.load_state_dict(state['model_state_dict'])
    model.eval()
    return model, state


@torch.no_grad()
def collect_predictions(model, loader, device='cpu'):
    model.eval()
    records = []
    for batch in loader:
        logits = model(batch['x'].to(device)).squeeze(-1)
        scores = logits.sigmoid().cpu().tolist()
        if not torch.isfinite(logits).all():
            raise ValueError('Nonfinite model output')
        records.extend({'sample_id': sid, 'device': d, 'label': int(y), 'fall_score': score}
                       for sid, d, y, score in zip(batch['sample_id'], batch['device'], batch['y'].tolist(), scores))
    if len(records) != len(loader.dataset):
        raise ValueError('Incomplete evaluation coverage')
    return records


def record_metrics(records, threshold):
    return metrics([r['label'] for r in records], [r['fall_score'] for r in records], threshold)
