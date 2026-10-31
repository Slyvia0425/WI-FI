
import os
from concurrent.futures import ProcessPoolExecutor
import joblib,numpy as np
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

def fit(name):
    import opt_v2_train as tr
    from opt_v2_common import OUT2,layout_ids
    target=OUT2/(name+'.joblib')
    if target.exists():return name+' cached'
    folder=OUT2/'heads_work'/name;folder.mkdir(exist_ok=True,parents=True);tr.OUT2=folder
    with np.load(OUT2/'features_train.npz') as a:X=a['X'];y=a['y']
    with np.load(OUT2/'features_validation.npz') as a:V=a['X'];vy=a['y']
    scaler=StandardScaler().fit(X);X=scaler.transform(X);V=scaler.transform(V)
    with threadpool_limits(limits=2):
        if name=='count':model=tr.fit_outer(name,X,y.sum(1).astype(int)-1,V,vy.sum(1).astype(int)-1,'count',82,(192,96))
        else:model=tr.fit_outer(name,X,layout_ids(y),V,vy,'layout',83,(512,256))
    if not target.exists():
        tmp=target.with_suffix('.worker.partial');joblib.dump(model,tmp);tmp.replace(target)
        trace=folder/(name+'_trace.json');(OUT2/trace.name).write_bytes(trace.read_bytes())
    return name+' complete'

if __name__=='__main__':
    with ProcessPoolExecutor(max_workers=2) as pool:print(list(pool.map(fit,['count','layout'])),flush=True)
