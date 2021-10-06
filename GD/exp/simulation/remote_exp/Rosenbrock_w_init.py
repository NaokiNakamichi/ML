import os, sys

import numpy as np

from ML2_lib import loss
from ML2_lib import RV_SGDAve

import pandas as pd
import datetime

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
    f_E_var = 1.75
    noise_type_f = "lognormal"
    w_init_list_0 = range(-20,20,2)
    w_init_list_1 = range(-20,20,2)

    k_string = [f"{i + 1}" for i in k_list]

    now = datetime.datetime.now()

    new_dir_path_recursive = f"remote_save_result/w_init_Rosenbrock_lognormal{now:%m:%d:%H:%M:%S}"
    os.makedirs(new_dir_path_recursive)

    for w_init_0 in range(-20,20,2):
        for w_init_1 in range(-20,20,2):

            son = loss.RosenBrock(d=d, noise_type=noise, E_var=E_var, f_E_var=f_E_var, noise_type_f=noise_type_f)
            RV = RV_SGDAve.RVSGDByW(model_opt=son, c=c, n=n, lr=lr)
            w_init = np.array([w_init_0,w_init_1])
            _, result = RV.many_trails(trial_num=trial_num, max_k=k_list[-1] + 1, w_init=w_init)
            title = f"RVSGD Rosenbrock trial = {trial_num} noise = {noise} D = {d}_sample_num{n} noise var = {E_var}　w_init = {w_init}"
            df = pd.DataFrame(result[:, k_list, 0], columns=k_string)
            df.to_csv(
                f"{new_dir_path_recursive}/noise_{noise}_trial_num_{trial_num}_D{d}_sample_num{n}_RV_w_init_{w_init}.csv",
                index=False)
