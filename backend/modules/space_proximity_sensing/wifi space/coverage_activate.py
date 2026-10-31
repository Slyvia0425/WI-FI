
import json
import numpy as np
from coverage_common import ROOT,DATA,OUT,valid_placement

def main():
    r=json.loads((OUT/'results.json').read_text(encoding='utf-8'));rows=json.loads((DATA/'manifest.json').read_text())
    if not r['activation_recommended']:raise RuntimeError('Validation improvement gate failed; the old model remains active. Inspect results before changing the protocol.')
    if any(r['duplicate_train_raw_channels'].values()):raise RuntimeError('Split leakage')
    if sum(r['split_counts'].values())!=len(rows):raise RuntimeError('Inconsistent report')
    if any(not valid_placement(a['xy'],a['device_offset_xy']) for a in rows):raise RuntimeError('Invalid physical placement')
    if len(list((DATA/'records').glob('*.npz')))!=len(rows):raise RuntimeError('Incomplete signal records')
    p=ROOT/'active_experiment.json';tmp=p.with_suffix('.partial');tmp.write_text(json.dumps(dict(experiment='coverage_v3',protocol=3,reason='Passed the predeclared validation improvement gate and placement/split checks',validation_gain=r['validation_gain']),indent=2),encoding='utf-8');tmp.replace(p)
    print('Full-coverage model activated. Reload http://127.0.0.1:8766/',flush=True)

if __name__=='__main__':main()
