
import json
import numpy as np
import joblib
from simulate import build_scene,response,ROOT
from learning import features,observe
from sklearn.metrics import accuracy_score

def main():
    scene,obj,solver=build_scene(device_shift=True);obj.position=[3,3,-10]
    ref=response(scene,solver,17)
    np.save(ROOT/'data/device_empty_reference.npy',ref)
    rows=[r for r in json.loads((ROOT/'data/manifest.json').read_text()) if r['split']=='device_shift']
    model=joblib.load(ROOT/'outputs/model.joblib')['model'];X=[];y=[]
    for row in rows:
        with np.load(ROOT/'data/records'/(row['id']+'.npz'),allow_pickle=False) as a:h=a['h']
        X.append(features(observe(h,np.random.default_rng(123)),ref));y.append(row['zone'])
    print('Shifted-device accuracy with a fresh empty-room reference:',accuracy_score(y,model.predict(X)),flush=True)

if __name__=='__main__':main()
