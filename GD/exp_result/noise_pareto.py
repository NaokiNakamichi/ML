import numpy as np
import random
import datetime
import pandas as pd
from tqdm import tqdm

import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import model_opt
import algo_GD
import helper
import noise

if __name__ == "__main__":
    args = sys.argv
    t = int(args[1])
    w_init = np.array([3, 3])
    _t_max = 3000
    f = model_opt.RosenBrock()
    w_star = f.w_star

    last_w_store = []
    iqr_store = []
    for i in tqdm(range(t)):
        a = 5
        noise_data = noise.Pareto(dim=2, n=_t_max, a=a).generate() * 500
        iqr = helper.iqr(noise_data)
        algo = algo_GD.SGD(w_init=w_init, t_max=_t_max, a=0.00078)
        for i in algo:
            noise_value = noise_data[algo.t - 1]
            f = model_opt.RosenBrock(noise_value=noise_value)
            algo.update(model=f)
        last_w_store.append(algo.w)
        iqr_store.append(iqr)

    dt_now = datetime.datetime.now()
    last_w_store = np.array(last_w_store)
    data = np.array([iqr_store,last_w_store[:,0],last_w_store[:,1]]).T
    df = pd.DataFrame(data=data, columns=['iqr', 'w_0', 'w_1'])
    df.to_csv('pareto_noise/pareto_noise{}_last_w{}.csv'.format(t,dt_now),header=True)