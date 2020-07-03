import numpy as np


def gauss(w, mean=0, sigma=1):
    np.random.seed()
    return np.random.normal(loc=mean, scale=sigma, size=w.shape)


def log_normal(w, mean=0, sigma=1):
    np.random.seed()
    return np.random.lognormal(mean=mean, sigma=sigma, size=w.shape)


def pareto(w, a=1):
    np.random.seed()
    return np.random.pareto(*w.shape, a)


def student_t(w,v):
    np.random.seed()
    return np.random.standard_t(v, size=w.shape)


def iqr(x):
    q75, q25 = np.percentile(x, [75, 25])
    iqr_value = q75 - q25
    return iqr_value

