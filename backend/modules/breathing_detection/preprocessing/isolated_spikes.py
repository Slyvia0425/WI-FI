"""Small fixed time-aware isolated-spike filter for offline experiments only."""
import numpy as np

HALF_WINDOW_SECONDS = .10
MAX_RUN = 2
MAX_BRACKET_SECONDS = .15


def resample_amplitudes(time, values, targets):
    time,values,targets = np.asarray(time,dtype=float),np.asarray(values,dtype=float),np.asarray(targets,dtype=float)
    if (time.ndim!=1 or targets.ndim!=1 or values.ndim!=2 or len(time)!=len(values) or len(time)<2
            or not np.isfinite(time).all() or not np.isfinite(values).all() or not np.isfinite(targets).all()
            or not np.all(np.diff(time)>0) or not np.all(np.diff(targets)>0)
            or not len(targets) or targets[0]<time[0] or targets[-1]>time[-1]):
        raise ValueError('Require ordered finite, aligned and covered amplitude samples/targets')
    return np.asarray([np.interp(targets,time,values[:,k]) for k in range(values.shape[1])],dtype=np.float32)[None]


def isolated_spike_filter(time,values,threshold=8.):
    time,values = np.asarray(time,dtype=float),np.asarray(values,dtype=float)
    if (time.ndim!=1 or values.ndim!=2 or len(time)!=len(values) or len(time)<5 or not values.shape[1]
            or not np.isfinite(time).all() or not np.isfinite(values).all() or not np.all(np.diff(time)>0)
            or threshold not in (6.,8.)):
        raise ValueError('Require finite ordered [time,K] samples and fixed threshold 6 or 8')
    medians = np.median(values,axis=0)
    global_mad = np.median(abs(values-medians),axis=0)
    flags = np.zeros(values.shape,dtype=bool)
    for i,t in enumerate(time):
        lo = np.searchsorted(time,t-HALF_WINDOW_SECONDS-1e-9,side='left')
        hi = np.searchsorted(time,t+HALF_WINDOW_SECONDS+1e-9,side='right')
        if hi-lo<5:
            continue
        local = values[lo:hi]
        median = np.median(local,axis=0)
        mad = np.median(abs(local-median),axis=0)
        scale = 1.4826*np.maximum(np.maximum(mad,.05*global_mad),1e-8)
        flags[i] = abs(values[i]-median)>threshold*scale
    result = values.copy()
    replaced = np.zeros_like(flags)
    for k in range(values.shape[1]):
        edges = np.diff(np.r_[False,flags[:,k],False].astype(int))
        for start,end in zip(np.flatnonzero(edges==1),np.flatnonzero(edges==-1)):
            left,right = start-1,end
            if (end-start>MAX_RUN or left<0 or right>=len(time)
                    or time[right]-time[left]>MAX_BRACKET_SECONDS+1e-9
                    or np.any(np.diff(time[left:right+1])>.1+1e-9)):
                continue
            result[start:end,k] = np.interp(time[start:end],time[[left,right]],values[[left,right],k])
            replaced[start:end,k] = True
    return result, {'flagged_cells':int(flags.sum()),'replaced_cells':int(replaced.sum()),
                    'replaced_frames':int(replaced.any(axis=1).sum()),'total_cells':int(values.size),
                    'replaced_fraction':float(replaced.mean())}


def preprocessing_configs():
    return [dict(name=f'{placement}_hampel{int(threshold)}',placement=placement,threshold=threshold)
            for placement in ('raw','resampled') for threshold in (6.,8.)]
