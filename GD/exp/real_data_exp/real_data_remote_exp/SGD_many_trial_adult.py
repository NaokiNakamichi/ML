# %%

import numpy as np
import pandas as pd

from ML2_lib import many_trials_SGD, many_trials_SGDAve
from ML2_lib import models

import datetime

if __name__ == "__main__":
    trial_num = 25

    name_path = "adult/adult_classification.names"
    train_path = "adult/adult_classification.csv"
    test_path = "adult/adult_classification.test"
    w_dim = 104
    class_num = 2
    unit_num = 4
    model_1 = models.LinearClassification(w_num=w_dim, c_num=class_num)
    model_2 = models.FF_L1(w_num=w_dim, c_num=class_num, unit_num=unit_num)
    model_3 = models.FF_L2(w_num=w_dim, c_num=class_num, unit_num=unit_num)
    model_list = [model_1, model_2, model_3]
    model_name_list = ["Linear", "FF_L1", "FF_L2"]
    lr_list = [0.0001, 0.001, 0.01, 0.1]
    many_learner = many_trials_SGD.ExManyTrials(train_path=train_path, name_path=name_path, test_path=test_path)

    result_accuracy, result_test_loss, col_name_list = many_learner.ex_many_settings(trial_num=trial_num,
                                                                                     model_name_list=model_name_list,
                                                                                     lr_list=lr_list,
                                                                                     model_list=model_list)

    hoge_SGD = pd.DataFrame(np.array(result_accuracy).T, columns=col_name_list)

    now = datetime.datetime.now()

    hoge_SGD.to_csv(f"SGD_adult_trial_{trial_num}_{now}.csv",index=False)

    many_learnerAve = many_trials_SGDAve.ExManyTrials(train_path=train_path, name_path=name_path, test_path=test_path)

    result_accuracy, _, col_name_list = many_learnerAve.ex_many_settings(trial_num=trial_num, model_name_list=model_name_list,
                                                                      lr_list=lr_list, model_list=model_list)

    hoge_SGDAve = pd.DataFrame(np.array(result_accuracy).T, columns=col_name_list)

    hoge_SGDAve.to_csv(f"SGDAve_adult_trial_{trial_num}_{now}.csv",index=False)




