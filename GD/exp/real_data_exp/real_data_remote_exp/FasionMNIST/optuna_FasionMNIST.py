import optuna

from torchvision.datasets import MNIST
from ML2_lib import format_data
import sys
import joblib
import datetime

import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn

from ML2_lib import models
from ML2_lib import SGDByTorch
import torchvision.datasets as datasets

if __name__ == "__main__":
    n_trials = int(sys.argv[1])
    data_transform = transforms.ToTensor()

    # 学習データを読み込む DataLoader を作成する。
    train_dataset = datasets.FashionMNIST(
        root="./data", train=True, transform=data_transform, download=True
    )
    train_data_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=1, shuffle=True
    )

    # テストデータを読み込む DataLoader を作成する。
    test_dataset = datasets.FashionMNIST(
        root="./data", train=False, transform=data_transform, download=True
    )
    test_data_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=1, shuffle=True
    )

    loss_type = nn.CrossEntropyLoss()

    model = models.CNNFAsionMNIST()


    def f(lr):
        hoge = SGDByTorch.SGDFromTrainLorder(lr=lr)
        hoge.learn(train_loader=train_data_loader, model=model, loss_type=loss_type)
        acc = hoge.test_accuracy(test_loader=test_data_loader)
        return acc


    def objective(trial: optuna.trial):
        lr = trial.suggest_loguniform("lr", 1e-7, 1e0)
        ret = f(lr)

        return ret


    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)

    now = datetime.datetime.now()

    joblib.dump(study, f"result_optuna/study_FASIONMNIST_{now}.pkl")
