import numpy as np
import random

def gauss(w,mean=0, sigma=1):
    np.random.seed()
    return np.random.normal(loc=mean,scale=sigma,size=w.shape)

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


def grad_norm(model,a=-1,b=1,n=100):
    norms = []
    for i in range(n):
        w = (b - a) * np.random.rand(2) + a
        g = model.g_opt(w)
        g_norm = np.linalg.norm(g, ord=2)
        norms.append(g_norm)
    norm = np.mean(norms)
    return norm

def var_random_noise(model,max_sigma=10):
    np.random.seed()
    sigma = (max_sigma - 1) * np.random.rand(1) + 1
    model(w)
    
def get_index_bins(w,a,b=0):
    return np.where((b < w) & (w < a))[0]

    
    
    