import numpy as np
import random
import multiprocessing as mproc
import matplotlib.pyplot as plt
import tqdm
from tqdm.notebook import tqdm as tqdm
import matplotlib.colors as colors
from matplotlib.cm import ScalarMappable

import model_opt
import algo_GD
import helper

if __name__ == "__main__":
    w_init = np.array([3,3])
    _t_max = 3000
    f = model_opt.RosenBrock()


    w_star = f.w_star

    last_w_store = []
    iqr_store = []
    for i in tqdm(range(10)):
        var = np.random.randint(1,300,1)[0]
        noise = helper.gauss
        f = model_opt.RosenBrock(noise=noise,var=var)
        algo = algo_GD.SGD(w_init=w_init,t_max=_t_max,a=0.00078)
        for i in algo:
            algo.update(model=f)

        iqr_store.append(helper.iqr(algo.noise_store))
        last_w_store.append(algo.w)