import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import datetime
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.colors as colors
from matplotlib.cm import ScalarMappable

from ML2_lib import RV_SGDAve
from ML2_lib import loss
from ML2_lib import plot_set


def n_exp(d, trial_num, lr, c, noise, E_var, w_init, k_list, n_list):
    son = loss.RosenBrock(d=d, noise_type=noise, E_var=E_var)
    k_string = [f"{i + 1}" for i in k_list]

    for n in n_list:
        RV = RV_SGDAve.RVSGDByW(model_opt=son, c=c, n=n, lr=lr)
        _, result = RV.many_trails(trial_num=trial_num, max_k=10, w_init=w_init)
        title = "f"
        plot_set.box_plot_k(result, k_list, k_string, title)

        now = datetime.datetime.now()
        df = pd.DataFrame(result[:, k_list], columns=k_string).describe()
        df.to_csv(f"save_result_data/{now:%m月%d日%H:%M:%S}_noise_{noise}_trial_num_{trial_num}_D{d}_sample_num{n}.csv")


def e_var_exp(d, trial_num, lr, c, noise, E_var_list, w_init, k_list, n):

    k_string = [f"{i + 1}" for i in k_list]

    for E_var in E_var_list:
        son = loss.RosenBrock(d=d, noise_type=noise, E_var=E_var)
        RV = RV_SGDAve.RVSGDByW(model_opt=son, c=c, n=n, lr=lr)
        _, result = RV.many_trails(trial_num=trial_num, max_k=10, w_init=w_init)
        title = "f"
        plot_set.box_plot_k(result, k_list, k_string, title)

        now = datetime.datetime.now()
        df = pd.DataFrame(result[:, k_list], columns=k_string).describe()
        df.to_csv(f"save_result_data/{now:%m月%d日%H:%M:%S}_noise_{noise}_trial_num_{trial_num}_D{d}_sample_num{n}.csv")


def d_exp(d_list, trial_num, lr, c, noise, E_var, w_init, k_list, n):
    k_string = [f"{i + 1}" for i in k_list]

    for d in d_list:
        w_init = np.full(d,w_init[0])
        son = loss.RosenBrock(d=d, noise_type=noise, E_var=E_var)

        RV = RV_SGDAve.RVSGDByW(model_opt=son, c=c, n=n, lr=lr)
        _, result = RV.many_trails(trial_num=trial_num, max_k=10, w_init=w_init)
        title = "f"
        plot_set.box_plot_k(result, k_list, k_string, title)

        now = datetime.datetime.now()
        df = pd.DataFrame(result[:, k_list], columns=k_string).describe()
        df.to_csv(f"save_result_data/{now:%m月%d日%H:%M:%S}_noise_{noise}_trial_num_{trial_num}_D{d}_sample_num{n}.csv")
