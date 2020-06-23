import numpy as np
import random

def gauss(w,loc=0, scale=1):
    np.random.seed()
    return np.random.normal(loc=loc,scale=scale,size=w.shape)

def lognormal(w,mean=0,sigma=1):
    np.random.seed()
    return np.random.lognormal(mean=mean,sigma=sigma,size=w.shape)

def pareto(w,a=1):
    np.random.seed()
    return np.random.pareto(*w.shape,a)

def iqr(x):
    q75, q25 = np.percentile(x, [75 ,25])
    iqr = q75 - q25
    return iqr

def signal_noise():
    pass

def g_norm()
    
    