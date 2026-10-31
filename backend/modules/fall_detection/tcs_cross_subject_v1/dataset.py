"""Memory-mapped GCV features; training-fitted normalization is optional."""
import json
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset


class GCVDataset(Dataset):
    def __init__(self, root, role, include_continuous=True, normalization=None):
        root = Path(root)
        self.summary = json.loads((root / 'preparation.json').read_text())
        self.rows = json.loads((root / 'samples.json').read_text())
        self.features = np.memmap(root / 'features.f32', dtype=np.float32, mode='r', shape=tuple(self.summary['shape']))
        self.indices = [i for i,r in enumerate(self.rows) if r['role']==role and (include_continuous or r['kind']=='single_action')]
        if not self.indices:
            raise ValueError('Empty requested dataset; test features are intentionally not prepared')
        self.mean = self.scale = None
        if normalization:
            cfg = json.loads(Path(normalization).read_text())
            if cfg['fit_role']!='train' or cfg['feature_cache_sha256']!=self.summary['feature_cache_sha256']:
                raise ValueError('Normalization source or data version mismatch')
            self.mean=np.asarray(cfg['mean'],dtype=np.float32)
            self.scale=np.asarray(cfg['scale'],dtype=np.float32)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, item):
        i=self.indices[item]
        x=np.array(self.features[i],copy=True)
        if self.mean is not None:
            x=(x-self.mean)/self.scale
        if x.shape!=(1,400,90) or not np.isfinite(x).all():
            raise ValueError('Invalid input feature')
        return torch.from_numpy(x), torch.tensor(self.rows[i]['label'],dtype=torch.long), self.rows[i]['sample_id']
