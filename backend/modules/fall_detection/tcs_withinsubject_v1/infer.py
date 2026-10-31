"""Independent-process single-SA-record inference for the TCS subject01 pilot."""
import argparse
import hashlib
import json
from pathlib import Path
import zipfile

import numpy as np
import torch

from model import CNN
from prepare import MODULE, sha, transform_dat_bytes


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--checkpoint', type=Path, required=True)
    ap.add_argument('--archive', type=Path, required=True)
    ap.add_argument('--member', required=True)
    ap.add_argument('--backend', choices=['cpu', 'mps'], default='cpu')
    ap.add_argument('--output', type=Path)
    args = ap.parse_args()
    torch.set_num_threads(2)
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    if checkpoint['label_mapping'] != {'Nonfall': 0, 'Fall': 1}:
        raise ValueError('Unexpected label mapping')
    cfg = checkpoint['run_config']
    for name in ('model.py', 'prepare.py'):
        if sha(Path(__file__).parent / name) != cfg['source_sha256'][name]:
            raise ValueError('Inference source differs from saved preprocessing/model source')
    if sha(MODULE / 'tcs_fall_audit_v1/audit_local_archives.py') != cfg['native_decoder_sha256']:
        raise ValueError('Native decoder source changed')
    # Avoid introducing a new data domain or evaluating the reserved continuous recordings.
    if args.member not in checkpoint['split_identifiers']['train_ids'] + checkpoint['split_identifiers']['validation_ids']:
        raise ValueError('This pilot command accepts its audited single-action IDs only; continuous inference is not configured')
    with zipfile.ZipFile(args.archive) as archive:
        data = archive.read(args.member)
    known = json.loads((MODULE / 'reports/tcs_fall_local_v1/records.json').read_text())
    audited = next(r for r in known if r['archive_member'] == args.member)
    if hashlib.sha256(data).hexdigest() != audited['raw_file_sha256']:
        raise ValueError('Raw record differs from audited data')
    x, preprocessing = transform_dat_bytes(data)
    model = CNN().to(args.backend).eval()
    model.load_state_dict(checkpoint['model_state_dict'], strict=True)
    with torch.inference_mode():
        score = model(torch.from_numpy(x[None]).to(args.backend)).softmax(1)[0, 1].item()
    result = {'archive': str(args.archive.resolve()), 'member': args.member,
              'fall_score': score, 'threshold': checkpoint['threshold'],
              'prediction': 'Fall' if score >= checkpoint['threshold'] else 'Nonfall',
              'model_version': checkpoint['model_version'], 'best_epoch': checkpoint['best_epoch'],
              'input_shape': [1, 1, 400, 90], 'backend': args.backend,
              'preprocessing': preprocessing, 'checkpoint_sha256': sha(args.checkpoint),
              'scope': 'single-subject single-action pilot; score not calibrated; no continuous alarm claim'}
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
