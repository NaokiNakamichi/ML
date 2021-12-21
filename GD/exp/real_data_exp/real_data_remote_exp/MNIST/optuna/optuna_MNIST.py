# %%

import optuna

from ML2_lib import SGDByTorch
from torchvision.datasets import MNIST
from ML2_lib import models
from ML2_lib import format_data
import sys
import joblib
import datetime

if __name__ == "__main__":

    n_trials = int(sys.argv[1])
    early_stopping = int(sys.argv[2])
    model_type = sys.argv[3]

    mnist_data = MNIST('data/MNIST', download=True, )

    # X_train = mnist_data.train_data.reshape(-1,784)
    # X_test = mnist_data.test_data.reshape(-1,784)
    # y_train = mnist_data.train_labels
    # y_test = mnist_data.test_labels

    X_train, y_train, X_test, y_test = format_data.MNIST_data().return_data()

    w_dim = X_train.shape[1]
    class_num = int(max(y_train) + 1)
    unit_num = 5

    if "linear" == model_type:
        model = models.LinearClassification(w_num=w_dim, c_num=class_num)

    elif "FF_L1" == model_type:
        model = models.FF_L1(w_num=w_dim, c_num=class_num, unit_num=unit_num)

    elif "FF_L2" == model_type:
        model = models.FF_L2(w_num=w_dim, c_num=class_num, unit_num=unit_num)

    else:
        raise ValueError("第二変数を確認、linear or FF_L1 or FF_L2")


    def f(lr):
        hoge = SGDByTorch.SGDTorch(lr=lr)
        _, acc = hoge.learn(x=X_train, y=y_train, model=model, class_num=class_num, X_test=X_test, Y_test=y_test,
                            early_stopping=early_stopping)
        return acc


    def objective(trial: optuna.trial):
        lr = trial.suggest_loguniform("lr", 1e-7, 1e0)
        ret = f(lr)

        return ret


    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)

    now = datetime.datetime.now()

    joblib.dump(study, f"result_optuna/study_MNIST_{model_type}_{now}.pkl")
