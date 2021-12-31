from ML2_lib import format_data
from ML2_lib import RV_SGD_Torch

from sklearn.model_selection import train_test_split

import joblib

import datetime

from tqdm import tqdm

import pandas as pd
import numpy as np

import sys

if __name__ == "__main__":
    n_trials = int(sys.argv[1])

    X_train, y_train, X_test, y_test = format_data.FashionMNIST_data().return_data()
    X_train, X_valid = train_test_split(X_train, test_size=0.1, shuffle=False)
    y_train, y_valid = train_test_split(y_train, test_size=0.1, shuffle=False)

    model_type_list = ["FF_L2","FF_L1","linear"]

    max_k = 10

    lr_dic = {"FF_L2": [0.0001, 0.0002, 0.0003, 0.0005], "FF_L1": [0.001,0.003,0.005,0.01], "linear": [0.01,0.03,0.05,0.1]}

    for model_type in tqdm(model_type_list):
        lr_list = lr_dic[model_type]
        for lr in lr_list:

            result_loss = []
            result_accuracy = []

            col_name_list = []

            for k in range(1, max_k):
                col_name_list.append(f"k_{k}")

            for i in tqdm(range(n_trials)):

                trial_result_loss = []
                trial_result_accuracy = []

                hoge = RV_SGD_Torch.RVSGDByTorch(lr=lr)
                k_model_list = []

                for k in range(1, max_k):
                    result_model = hoge.run_RVSGD(x_train=X_train, y_train=y_train, valid_x=X_valid, valid_y=y_valid,
                                                  k=k,
                                                  model_type=model_type)
                    k_model_list.append(result_model)

                for i_model in k_model_list:
                    test_loss, accuracy = hoge.prediction(x=X_test, y=y_test, model=i_model, is_print=False)
                    trial_result_accuracy.append(accuracy)
                    trial_result_loss.append(test_loss)

                result_loss.append(trial_result_loss)
                result_accuracy.append(trial_result_accuracy)

            now = datetime.datetime.now().strftime('%Y年%m月%d日%H時%M分')

            result_accuracy_pd = pd.DataFrame(np.array(result_accuracy), columns=col_name_list)

            result_accuracy_pd.to_csv(f"result_/RVSGD_FasionMNIST_trial_{n_trials}_{now}_model_{model_type}_lr_{lr}.csv",
                                      index=False)