import numpy as np
import pickle
import os, sys
from ML2_lib import loss
from ML2_lib import RV_SGDAve
import datetime
from tqdm import tqdm

if __name__ == "__main__":

    d = 2
    lr = 0.001
    c = 1
    n = 10000
    k_list = [1, 2, 4, 5, 10, 20]

    # ノイズの種類（勾配）
    noise_type_list = ["lognormal", "normal", "student_t"]

    E_var = 1.75


    # ノイズの分散（勾配）
    def noise_var(noise_type):
        if noise_type == "lognormal":
            return [1.2, 1.75, 1.9]
        elif noise_type == "normal":
            return [1.9]
        elif noise_type == "student_t":
            return [1.5, 3]
        else:
            raise ValueError("lognormalかnormalかsutdent_tで")


    # 検証時に加える関数値にノイズの種類
    noise_type_f = "lognormal"
    f_E_var = 1.75

    noise = "lognormal"

    w_init = np.array([2, 2])

    # 初期値
    w_init_list_0 = [-4, -2, 0, 2, 4]
    w_init_list_1 = [-4, -2, 0, 2, 4]

    k_string = [f"{i + 1}" for i in k_list]

    now = datetime.datetime.now()

    new_dir_path_recursive = f"remote_save_result/Rosenbrock_2d_trajectory{now:%m:%d:%H:%M:%S}"
    os.makedirs(new_dir_path_recursive)

    for noise_type in tqdm(noise_type_list):
        noise_var_list = noise_var(noise_type)
        for E_var in tqdm(noise_var_list):
            for w_init_0 in w_init_list_0:
                for w_init_1 in w_init_list_1:
                    result_to_pickle = {}
                    son = loss.RosenBrock(d=d, noise_type=noise_type, E_var=E_var, f_E_var=f_E_var,
                                          noise_type_f=noise_type_f)
                    RV = RV_SGDAve.RVSGDByW(model_opt=son, c=c, n=n, lr=lr)
                    w_init = np.array([w_init_0, w_init_1])
                    k_core_list, k_selected_index, k_w_rv = RV.k_all_transition(k_list=k_list, w_init=w_init)
                    result_to_pickle["k_core_list"] = k_core_list
                    result_to_pickle["k_selected_index"] = k_selected_index
                    title = f"RVSGD_Rosenbrock_trajectory_noise_{noise_type}_D_{d}_sample_num{n}_noise_va_{E_var}_w_init_{w_init_0}_{w_init_1}f_noise{noise_type_f}f_noise_E{f_E_var}.pickle"
                    with open(f"{new_dir_path_recursive}/{title}", mode="wb") as f:
                        pickle.dump(result_to_pickle, f)
