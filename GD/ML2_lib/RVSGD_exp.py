import pandas as pd
import numpy as np
import datetime
from tqdm import tqdm
import os

from ML2_lib import RV_SGDAve
from ML2_lib import loss
from ML2_lib import plot_set


def easy_exp(trial_num, lr, c, w_init, k_list, n, son, title="", folder_title="",saving_png = False):
    k_string = [f"{i + 1}" for i in k_list]
    RV = RV_SGDAve.RVSGDByW(model_opt=son, c=c, n=n, lr=lr)
    _, result = RV.many_trails(trial_num=trial_num, max_k=k_list[-1] + 1, w_init=w_init)
    plot_set.box_plot_k(result, k_list, k_string, title=title, folder_title=folder_title, saving_png=saving_png)


def n_exp(d, trial_num, lr, c, noise, E_var, w_init, k_list, n_list):
    son = loss.RosenBrock(d=d, noise_type=noise, E_var=E_var)
    k_string = [f"{i + 1}" for i in k_list]

    for n in n_list:
        RV = RV_SGDAve.RVSGDByW(model_opt=son, c=c, n=n, lr=lr)
        _, result = RV.many_trails(trial_num=trial_num, max_k=k_list[-1] + 1, w_init=w_init)
        title = "f"
        plot_set.box_plot_k(result, k_list, k_string, title)

        now = datetime.datetime.now()
        df = pd.DataFrame(result[:, k_list], columns=k_string)
        df.to_csv(
            f"save_result_data/{now:%m月%d日%H:%M:%S}_noise_{noise}_trial_num_{trial_num}_D{d}_sample_num{n}_RV.csv")


def e_var_exp(d, trial_num, lr, c, noise, E_var_list, w_init, k_list, n, noise_type_f=None, f_E_var=1.75):
    k_string = [f"{i + 1}" for i in k_list]
    now = datetime.datetime.now()
    new_dir_path_recursive = f"remote_save_result/Rosenbrock_2d_grad_noise_student_t{now:%m:%d:%H:%M:%S}"
    os.makedirs(new_dir_path_recursive)

    for E_var in tqdm(E_var_list):
        son = loss.RosenBrock(d=d, noise_type=noise, E_var=E_var, noise_type_f=noise_type_f, f_E_var=f_E_var)
        RV = RV_SGDAve.RVSGDByW(model_opt=son, c=c, n=n, lr=lr)
        _, result = RV.many_trails(trial_num=trial_num, max_k=k_list[-1] + 1, w_init=w_init)
        # title = "f"
        # plot_set.box_plot_k(result, k_list, k_string, title)

        df = pd.DataFrame(result[:, k_list], columns=k_string)
        df.to_csv(
            f"{new_dir_path_recursive}/noise_{noise}_Evar{E_var}_trial_num_{trial_num}_D{d}_sample_num{n}_RV_w_init_{w_init[0]}_{w_init[1]}_f_noise{noise_type_f}_f_noise_E{f_E_var}_lr{lr}_n_{n}.csv",
            index=False)


def d_exp(d_list, trial_num, lr, c, noise, E_var, w_init, k_list, n):
    k_string = [f"{i + 1}" for i in k_list]

    for d in d_list:
        w_init = np.full(d, w_init[0])
        son = loss.RosenBrock(d=d, noise_type=noise, E_var=E_var)

        RV = RV_SGDAve.RVSGDByW(model_opt=son, c=c, n=n, lr=lr)
        _, result = RV.many_trails(trial_num=trial_num, max_k=k_list[-1] + 1, w_init=w_init)
        title = "f"
        plot_set.box_plot_k(result, k_list, k_string, title)

        now = datetime.datetime.now()
        df = pd.DataFrame(result[:, k_list], columns=k_string)
        df.to_csv(
            f"save_result_data/{now:%m月%d日%H:%M:%S}_noise_{noise}_trial_num_{trial_num}_D{d}_sample_num{n}_RV.csv")


def w_init_exp(d, trial_num, lr, c, noise, E_var, w_init_list, k_list, n, f_E_var=1.75, noise_type_f=None):
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
