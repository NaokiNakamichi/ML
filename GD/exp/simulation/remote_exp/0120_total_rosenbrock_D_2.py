import os, sys

import numpy as np

from ML2_lib import loss
from ML2_lib import RV_SGDAve

import pandas as pd
import datetime
from tqdm import tqdm

if __name__ == "__main__":
    args = sys.argv
    d = 2
    trial_num = 100
    # [2,2]では0.001
    lr = 0.0007
    c = 1
    n = 10000
    k_list = [1, 2, 4, 5, 10, 20]

    # ノイズの種類（勾配）
    # noise_type_list = ["lognormal", "normal", "student_t"]
    noise_type_list = ["lognormal"]

    E_var = 1.75


    # ノイズの分散（勾配）
    def noise_var(noise_type):
        if noise_type == "lognormal":
            # return [1.2, 1.75, 1.9]
            return [1.75]
        elif noise_type == "normal":
            return [1.9]
        elif noise_type == "student_t":
            return [1.5, 3]
        else:
            raise ValueError("lognormalかnormalかsutdent_tで")


    # 検証時に加える関数値にノイズの種類
    noise_type_f = "normal"
    f_E_var = 1.75

    # 初期値
    w_init_list_0 = [3]
    w_init_list_1 = [3]

    k_string = [f"{i + 1}" for i in k_list]

    now = datetime.datetime.now()

    new_dir_path_recursive = f"remote_save_result/Rosenbrock_2d_total{now:%m:%d:%H:%M:%S}"
    os.makedirs(new_dir_path_recursive)

    for noise_type in tqdm(noise_type_list):
        noise_var_list = noise_var(noise_type)
        for E_var in noise_var_list:
            for w_init_0 in w_init_list_0:
                for w_init_1 in w_init_list_1:
                    son = loss.RosenBrock(d=d, noise_type=noise_type, E_var=E_var, f_E_var=f_E_var, noise_type_f=noise_type_f)
                    RV = RV_SGDAve.RVSGDByW(model_opt=son, c=c, n=n, lr=lr)
                    w_init = np.array([w_init_0, w_init_1])
                    _, result = RV.many_trails(trial_num=trial_num, max_k=k_list[-1] + 1, w_init=w_init)
                    title = f"RVSGD Rosenbrock trial = {trial_num} noise = {noise_type} D = {d}_sample_num{n} noise var = {E_var}　w_init = {w_init} f_noise{noise_type_f} f_noise_E{f_E_var}"
                    df = pd.DataFrame(result[:, k_list], columns=k_string)

                    df.to_csv(
                        f"{new_dir_path_recursive}/noise_{noise_type}_Evar{E_var}_trial_num_{trial_num}_D{d}_sample_num{n}_RV_w_init_{w_init[0]}_{w_init[1]}_f_noise{noise_type_f}_f_noise_E{f_E_var}_lr{lr}_n_{n}.csv",
                        index=False)
