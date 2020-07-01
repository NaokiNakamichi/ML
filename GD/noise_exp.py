import numpy as np
import random
import datetime
import pandas as pd
import sys
from tqdm import tqdm

import model_opt
import algo_GD
import helper

if __name__ == "__main__":
    args = sys.argv
    t = int(args[1])
    w_init = np.array([3,3])
    _t_max = 3000
    f = model_opt.RosenBrock()


    w_star = f.w_star

    last_w_store = []
    iqr_store = []
    for i in tqdm(range(t)):
        var = np.random.randint(1,300,1)[0]
        noise = helper.gauss
        f = model_opt.RosenBrock(noise=noise,var=var)
        algo = algo_GD.SGD(w_init=w_init,t_max=_t_max,a=0.00078)
        for i in algo:
            algo.update(model=f)

        iqr_store.append(helper.iqr(algo.noise_store))
        last_w_store.append(algo.w)
        
       
    
    dt_now = datetime.datetime.now()
    last_w_store = np.array(last_w_store)
    data = np.array([iqr_store,last_w_store[:,0],last_w_store[:,1]]).T
    df = pd.DataFrame(data=data, columns=['iqr', 'w_0', 'w_1'])
    df.to_csv('exp_result/gauss_noise_last_w  {}.csv'.format(dt_now),header=True)
