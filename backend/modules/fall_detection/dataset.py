"""One complete HDF5 segment per sample, using saved train-only preprocessing."""
import hashlib
from pathlib import Path

import torch
from torch.utils.data import Dataset

if __package__:
    from .preprocess import Preprocessor, manifest_rows
else:
    from preprocess import Preprocessor, manifest_rows


class FallDataset(Dataset):
    def __init__(self, manifest, preprocess_config, split):
        self.preprocessor = Preprocessor(preprocess_config)
        actual = hashlib.sha256(Path(manifest).read_bytes()).hexdigest()
        if actual != self.preprocessor.config['manifest_sha256']:
            raise ValueError('Manifest changed since normalization fitting; refit/review provenance')
        self.rows = [r for r in manifest_rows(manifest) if split in r['split'].split(';')]
        if not self.rows:
            raise ValueError(f'Empty or unknown split: {split}')

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        x, mask = self.preprocessor.transform_file(row['file_path'], row['device'])
        variable_mask = mask.copy()
        constant = self.preprocessor.config['devices'][row['device']]['constant_training_features']
        variable_mask[constant] = False
        return {'x': torch.from_numpy(x), 'y': torch.tensor(float(row['binary_label']), dtype=torch.float32),
                'feature_mask': torch.from_numpy(mask), 'variable_feature_mask': torch.from_numpy(variable_mask),
                'valid_features': int(mask.sum()),
                'sample_id': row['sample_id'], 'device': row['device']}
