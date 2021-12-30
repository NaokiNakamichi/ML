from ML2_lib import format_data
from ML2_lib import RV_SGD_Torch

from sklearn.model_selection import train_test_split

import joblib

import datetime

from tqdm import tqdm

import pandas as pd
import numpy as np
import torch

import torchvision
import torchvision.transforms as transforms

import sys

if __name__ == "__main__":
    n_trials = int(sys.argv[1])

    # ToTensor：画像のグレースケール化（RGBの0~255を0~1の範囲に正規化）、Normalize：Z値化（RGBの平均と標準偏差を0.5で決め打ちして正規化）
    # ToTensor：画像のグレースケール化（RGBの0~255を0~1の範囲に正規化）、Normalize：Z値化（RGBの平均と標準偏差を0.5で決め打ちして正規化）
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    # トレーニングデータをダウンロード
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)

    # テストデータをダウンロード
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=True, num_workers=2)

    n_samples = len(trainset)  # n_samples is 60000
    train_size = int(len(trainset) * 0.8)  # train_size is 48000
    val_size = n_samples - train_size  # val_size is 48000

    # shuffleしてから分割してくれる.
    train_dataset, val_dataset = torch.utils.data.random_split(trainset, [train_size, val_size])
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=2)

    max_k = 10

    lr_list = [0.0001, 0.0002, 0.0003, 0.0005, 0.001, 0.003, 0.005, 0.01]

    for lr in lr_list:
        result_accuracy = []

        col_name_list = []

        for k in range(1, max_k):
            col_name_list.append(f"k_{k}")

        for i in tqdm(range(n_trials)):

            trial_result_loss = []
            trial_result_accuracy = []

            hoge = RV_SGD_Torch.RVSGDByTorchWithLoader(lr=lr, train_loader=train_loader, test_loader=testloader,
                                                       valid_dataset=val_dataset.dataset, )
            k_model_list = []

            for k in range(1, max_k):
                result_model = hoge.run_RVSGD(k=k)
                k_model_list.append(result_model)

            for i_model in k_model_list:
                accuracy = hoge.prediction(model=i_model)
                trial_result_accuracy.append(accuracy)
                # trial_result_loss.append(test_loss)

            # result_loss.append(trial_result_loss)
            result_accuracy.append(trial_result_accuracy)

        now = datetime.datetime.now().strftime('%Y年%m月%d日%H時%M分')

        result_accuracy_pd = pd.DataFrame(np.array(result_accuracy), columns=col_name_list)

        result_accuracy_pd.to_csv(f"result_/RVSGD_CIFAR10_trial_{n_trials}_{now}_model_CNN_lr_{lr}.csv",
                                  index=False)
