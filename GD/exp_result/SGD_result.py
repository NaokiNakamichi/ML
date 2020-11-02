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

algo = SGD(w_init=w_init, t_max=_t_max, a=0.1)
for i in range(10):
    try:
        for i in algo:
            f = model_opt.Perm(noise_value=noise_value)
            algo.update(model=f)
    except RuntimeWarning:
        print("fuga")


