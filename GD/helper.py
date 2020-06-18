import numpy as np
import random

def gauss(w):
    np.random.seed()
    return np.random.randn(*w.shape)

def lognormal(w):
    np.random.seed()
    return np.random.lognormal(*w.shape)

def pareto(w,a=1):
    np.random.seed()
    return np.random.pareto(*w.shape,a)
    
    