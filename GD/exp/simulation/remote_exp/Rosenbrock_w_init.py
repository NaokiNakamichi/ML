import os, sys

sys.path.append(os.path.join(os.path.dirname(__file__), "..."))

import numpy as np

from ML2_lib import loss
from ML2_lib import plot_set
from ML2_lib import RVSGD_exp
from ML2_lib import RV_SGDAve

import pandas as pd
import datetime



if __name__ == "__main__":
    args = sys.argv
    d = 2
    trial_num = 5
    lr = 0.001

    c = 1

    n = 10000
    k_list = [0, 1, 3, 4, 9, 19]

    noise = "lognormal"
    E_var = 1.75
    f_E_var = 1.75
    noise_type_f = "lognormal"
    w_init_list = [np.array([0, i]) for i in range(5)]
    RVSGD_exp.w_init_exp(d=d, trial_num=trial_num, lr=lr, c=c, noise=noise, E_var=E_var, w_init_list=w_init_list,
                         k_list=k_list, n=n)

    son = loss.RosenBrock(d=d, noise_type=noise, E_var=E_var)
    k_string = [f"{i + 1}" for i in k_list]

    for w_init in w_init_list:
        son = loss.RosenBrock(d=d, noise_type=noise, E_var=E_var, f_E_var=f_E_var, noise_type_f=noise_type_f)
        RV = RV_SGDAve.RVSGDByW(model_opt=son, c=c, n=n, lr=lr)
        _, result = RV.many_trails(trial_num=trial_num, max_k=k_list[-1] + 1, w_init=w_init)
        title = f"RVSGD Rosenbrock trial = {trial_num} noise = {noise} D = {d}_sample_num{n} noise var = {E_var}　w_init = {w_init}"
        plot_set.box_plot_k(result, k_list, k_string, title)

        now = datetime.datetime.now()
        df = pd.DataFrame(result[:, k_list], columns=k_string)
        df.to_csv(
            f"remote_save_result/{now:%m月%d日%H:%M:%S}_noise_{noise}_trial_num_{trial_num}_D{d}_sample_num{n}_RV_w_init_{w_init[1]}.csv",
            index=False)
