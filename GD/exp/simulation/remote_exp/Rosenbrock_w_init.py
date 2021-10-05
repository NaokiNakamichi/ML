import numpy as np

from ML2_lib import loss
from ML2_lib import plot_set
from ML2_lib import RVSGD_exp

import pandas as pd
import datetime

import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

if __name__ == "__main__":
    args = sys.argv
    d = 2
    trial_num = 100
    lr = 0.001

    c = 1

    n = 10000
    k_list = [0, 1, 3, 4, 9, 19]



    noise = "lognormal"
    E_var = 1.75
    E_var = 1.75
    w_init_list = [np.array([0, i]) for i in range(5)]
    RVSGD_exp.w_init_exp(d=d, trial_num=trial_num, lr=lr, c=c, noise=noise, E_var=E_var, w_init_list=w_init_list,
                         k_list=k_list, n=n)
