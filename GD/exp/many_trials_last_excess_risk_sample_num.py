import pandas as pd
import numpy as np
import datetime

import os, sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from ML2_lib import DC_SGD
from ML2_lib import loss
from ML2_lib import plot_set


def main():
    args = sys.argv
    print(args)
    for i in args[1:]:
        noise_list = ["normal", "lognormal"]
        E_var_list = [2.2, 1.75]
        for noise, E_var in zip(noise_list, E_var_list):
            X_mean = 5
            X_var = 2

            w_star = np.array([1, 2, 3, 4, 5, 6])
            w_star = w_star.reshape(1, -1)
            n = int(i)
            print(n)
            c = 5
            son = loss.LinearQuadraticLoss()
            DC = DC_SGD.DCSGDSimulation(w_star=w_star, n=n, E_var=E_var, X_mean=X_mean, X_var=X_var, noise=noise,
                                        loss_type=son,
                                        c=c, fixed_lr=True)
            trial_num = 10
            _, result_loss_gauss = DC.many_trails(trial_num=trial_num, max_k=20)

            df = pd.DataFrame(result_loss_gauss)
            now = datetime.datetime.now()
            df.to_csv(
                f"GD/exp/simulation/save_result_data/{now:%m月%d日%H:%M:%S}_noise_{noise}_trial_num_{trial_num}_E_var{E_var}.csv")


if __name__ == "__main__":
    main()
