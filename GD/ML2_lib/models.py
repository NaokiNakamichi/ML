import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


class Model():
    def __init__(self, noise=None, name=""):
        self.name = name
        self.noise = noise

    def add_noise(self, w, sigma=1):
        if self.noise == None:
            return np.zeros(w.shape)
        return self.noise(w, sigma=sigma)


class LinearClassification(nn.Module):
    def __init__(self, w_num, c_num):
        super(LinearClassification, self).__init__()
        self.fc1 = nn.Linear(w_num, c_num)

    def forward(self, x):
        print(len(x.shape))

        if len(x.shape) == 1:
            x = torch.unsqueeze(x, 0)

        # print(f"front{x}")
        x = self.fc1(x)
        # print(f"rear{x}")
        return F.log_softmax(x, dim=1)

    def parameter_init(self):
        nn.init.constant_(self.fc1.weight, 0)
        nn.init.constant_(self.fc1.bias, 0)


class Net(nn.Module):
    # w_numは特徴量の次元数、c_numはクラス数、unit_numは中間層のユニット数
    def __init__(self, w_num, c_num, unit_num):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(w_num, unit_num)
        self.fc2 = nn.Linear(unit_num, unit_num)
        self.fc3 = nn.Linear(unit_num, c_num)
        self.relu = nn.ReLU()

    def forward(self, x):
        if len(x.shape) == 1:
            x = torch.unsqueeze(x, 0)

        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        # dimは0なら列単位でSoftmaxをかけてくれる。1なら行単位でSoftmaxをかけてくれる。
        return F.log_softmax(x, dim=1)

    # モデルパラメータを初期化
    def parameter_init(self):
        nn.init.uniform_(self.fc1.weight, -5, 5)
        nn.init.constant_(self.fc1.bias, 0)
        nn.init.uniform_(self.fc2.weight, -5, 5)
        nn.init.constant_(self.fc2.bias, 0)
        nn.init.uniform_(self.fc3.weight, -5, 5)
        nn.init.constant_(self.fc3.bias, 0)
