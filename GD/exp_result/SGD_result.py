import numpy as np
import random
import datetime
import pandas as pd
from tqdm import tqdm
import warnings

import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import model_opt
from algo_GD import SGD
import helper
import noise

"""
やりたいこと
各テスト関数おいての適切な学習率の測定
また初期値の違いでどうなるか
次元数が多い時の様子
ノイズの大きさによる
"""


warnings.resetwarnings()
warnings.simplefilter('error', RuntimeWarning)


w_init = np.array([2,-2])
noise_value = np.zeros(2)
var = 50 # iqr 70~80
batchsize = 100
_t_max = 100
noise_data = noise.Gauss(mean=0, sigma=var, dim=2, n=batchsize * _t_max).generate()
lr_list = [0.00001, 0.00005, 0.0001, 0.005, 0.001, 0.01, 0.1, 1]
model_list = [model_opt.Perm, model_opt.RosenBrock, model_opt.SumOfDifferent, model_opt.SumSquares]
noise_var_list = [0, 10, 30, 50, 70, 90, 100]
w_range = [-5,5]
for model in model_list:
    for var in noise_var_list:
        for lr in lr_list:
            for _ in range(100):

                w_init = (w_range[1] - w_range[0]) * np.random.rand(10000) + w_range[0]
                algo = SGD(w_init=w_init, t_max=_t_max, a=lr)
                try:
                    for _ in algo:
                        noise_value = noise.Gauss(mean=0, sigma=var, dim=10000, n=1).generate()
                        f = model(noise_value=noise_value)
                        algo.update(model=f)
                except RuntimeWarning:
                    print("fuga")




