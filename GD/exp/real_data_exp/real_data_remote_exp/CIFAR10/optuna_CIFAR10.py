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

if __name__ == "__main__":
    n_trials = int(sys.argv[1])
    # ToTensor：画像のグレースケール化（RGBの0~255を0~1の範囲に正規化）、Normalize：Z値化（RGBの平均と標準偏差を0.5で決め打ちして正規化）
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    # トレーニングデータをダウンロード
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=1, shuffle=True, num_workers=2)

    # テストデータをダウンロード
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=True, num_workers=2)

    model = models.CNN()
    loss_type = nn.CrossEntropyLoss()


    def f(lr):
        hoge = SGDByTorch.SGDFromTrainLorder(lr=lr)
        hoge.learn(train_loader=trainloader, model=model, loss_type=loss_type)
        acc = hoge.test_accuracy(test_loader=testloader)
        return acc


    def objective(trial: optuna.trial):
        lr = trial.suggest_loguniform("lr", 1e-7, 1e0)
        ret = f(lr)

        return ret


    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)

    now = datetime.datetime.now()

    joblib.dump(study, f"result_optuna/study_CIFAR10_{now}.pkl")
