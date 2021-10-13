import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from ML2_lib import RVSGD_exp

if __name__ == "__main__":

    d = 2
    trial_num = 100
    lr = 0.001
    noise = "student_t"
    c = 1
    w_init = np.array([2,2])
    n = 10000
    k_list = [0,1,3,4,9,19]
    E_var_list = [1.2,1.5,2,3]
    f_E_var = 1.75
    RVSGD_exp.e_var_exp(d=d,trial_num=trial_num,lr=lr,c=c,noise=noise,E_var_list=E_var_list,w_init=w_init,k_list=k_list,n=n,noise_type_f="lognormal",f_E_var=f_E_var)