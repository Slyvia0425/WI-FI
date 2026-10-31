
import numpy as np
from opt_v2_common import window as basic_window

def window(h,rng,snr_db=None,frames=16):
 if snr_db is None:snr_db=float(rng.uniform(25,40))
 a=basic_window(h,rng,snr_db,frames)
 f=np.linspace(-1,1,64)[None,None,:]
 phase=rng.normal(0,.025,(1,32,1))+rng.normal(0,.02,(1,32,1))*f+rng.normal(0,.008,(frames,1,1))
 return (a*np.exp(1j*phase)*10**(rng.normal(0,.15,(1,32,1))/20)).astype(np.complex64)
