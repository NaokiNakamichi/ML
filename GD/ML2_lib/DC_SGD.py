import numpy as np
import miniball

from tqdm.notebook import tqdm

from . import algo_sgd
from . import additive_noise
from . import merge

import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim


class DCSGD:
    def __init__(self, loss_type, c, lr=0.01, fixed_lr=True):
        self.loss_type = loss_type
        try:
            self.loss_type.type
        except:
            raise ValueError("損失を確認してください。")
        self.c = c
        # 学習率は固定ではない。
        self.lr = lr
        self.fixed_lr = fixed_lr

    def learn(self, k, w_init, x, y):

        model_store = []
        core_store = []

        for i in range(k):
            core_num = y.shape[0] // k
            data = [x[i:i+core_num], y[i:i+core_num]]
            # print(data)
            # print(data[0].shape)
            # print(data[1].shape)
            core = algo_sgd.SGD(w_init=w_init, a=self.lr, t_max=core_num - 1, data=data, fixed_lr=self.fixed_lr)
            for _ in core:
                core.update(self.loss_type)
            core_store.append(core.wstore)
            model_store.append(core.w)

        w_dc, _ = miniball.get_bounding_ball(np.array(model_store).reshape((-1, x.shape[1])), epsilon=1e-7)
        # w_dc = merge.median(np.array(model_store).reshape((-1, x.shape[1])))
        # print(w_dc)
        # print(model_store)
        w_dc = w_dc.reshape(1, -1)

        return w_dc, core_store


class DCSGDSimulation(DCSGD):
    def __init__(self, w_star, n, E_var, X_mean, X_var, noise, loss_type, c, lr=0.01, fixed_lr=True):
        super(DCSGDSimulation, self).__init__(loss_type, c, lr=lr, fixed_lr=True)
        try:
            self.w_star = w_star.reshape(1, -1)
        except:
            raise ValueError("w_starのデータ型を確認してください。numpy ndarrayのベクトルの形で入力")
        self.n = n
        self.X_mean = X_mean
        self.X_var = X_var
        self.E_var = E_var
        self.noise = noise
        self.loss_type = loss_type
        try:
            self.w_star = w_star.reshape(1, -1)
        except:
            raise ValueError("w_starのデータ型を確認してください。numpy ndarrayのベクトルの形で入力")
        try:
            self.loss_type.type
        except:
            raise ValueError("損失を確認してください。")
        self.c = c
        self.d = self.w_star.shape[1]
        self.lr = 0.01 / np.sqrt(self.d)
        # 学習率は固定かどうか選択する。
        self.fixed_lr = fixed_lr

    def generate_dataset(self):
        rng = np.random.default_rng()
        X = rng.normal(loc=self.X_mean, size=(self.n, self.d), scale=self.X_var)
        tmp = additive_noise.Noise(dim=1, mean=0, sigma=self.E_var, n=self.n)
        E = getattr(tmp, self.noise)()
        Y = np.dot(self.w_star, X.T) + E
        Y = Y.T
        return X, Y

    def trial_k(self, max_k):

        w_trial = []  # モデルを貯めていく、必要かどうか
        loss_store = []  # 過剰期待損失を貯めていく
        if max_k < 1:
            raise ValueError("max_k は0以上の整数")
        rng = np.random.default_rng()
        w_init = self.w_star + rng.uniform(-self.c, self.c, size=self.d)
        for k in range(1, max_k + 1):
            x, y = self.generate_dataset()
            w, _ = self.learn(k=k, w_init=w_init, x=x, y=y)
            w_trial.append(w)
            excess_risk = self.loss_type.excess_risk_normal(X_mean=self.X_mean, X_var=self.X_var, w_star=self.w_star,
                                                            w=w)
            loss_store.append(excess_risk)

        return np.array(w_trial), np.array(loss_store)

    def many_trails(self, trial_num, max_k):
        result_w = []  # パラメータの最終結果　トライアル数*分割数k*特徴量次元
        result_loss = []  # 過剰期待損失の最終結果　トライアル数*分割数k
        for _ in tqdm(range(trial_num)):
            w_trial, loss_store = self.trial_k(max_k=max_k)
            result_w.append(np.array(w_trial))
            result_loss.append(np.array(loss_store))
        return np.array(result_w), np.array(result_loss)

    def transition(self, k, w_init):

        x, y = self.generate_dataset()
        _, core_store = self.learn(k=k, w_init=w_init, x=x, y=y)
        w_transition = []
        loss_transition = []

        core_store = np.array(core_store)

        tmp = core_store.shape
        core_num, update_num, _, w_dim = tmp
        core_store = core_store.reshape(core_num, update_num, w_dim)
        core_store = core_store.transpose(1, 0, 2)
        for i in range(update_num):
            w, _ = miniball.get_bounding_ball(core_store[i, :, :])

            w_transition.append(w)
            loss_transition.append(
                self.loss_type.excess_risk_normal(X_mean=self.X_mean, X_var=self.X_var, w_star=self.w_star,
                                                  w=w.reshape(1, w_dim)))

        return w_transition, loss_transition


class DCSGDRealData(DCSGD):
    def __init__(self, loss_type, c, lr=0.01, fixed_lr=True):
        super(DCSGDRealData, self).__init__(loss_type, c, lr=lr, fixed_lr=True)

        self.loss_type = loss_type
        try:
            self.loss_type.type
        except:
            raise ValueError("損失を確認してください。")

        self.lr = lr
        # 学習率は固定かどうか選択する。
        self.fixed_lr = fixed_lr
        self.w = np.array(0)
        self.core_store = []

    def learn_data(self, k, w_init, x, y):
        self.w, self.core_store = self.learn(k, w_init, x, y)

    def predict(self, x):
        return np.dot(x, self.w.T)[0][0]

    def transition(self, x, y):

        w_transition = []
        loss_transition = []

        core_store = np.array(self.core_store)

        tmp = core_store.shape
        core_num, update_num, _, w_dim = tmp
        core_store = core_store.reshape(core_num, update_num, w_dim)
        core_store = core_store.transpose(1, 0, 2)
        for i in range(update_num):
            w, _ = miniball.get_bounding_ball(core_store[i, :, :])
            w = w.reshape(1, -1)

            w_transition.append(w)
            # print("x {}".format(x.shape))
            # print("y {}".format(y.shape))
            # print(self.loss_type.f(y=y, x=x, w=w.T))
            loss_transition.append(self.loss_type.f(y=y, x=x, w=w.T))

        return w_transition, loss_transition


class DCSGDByTorch:
    def __init__(self, loss, lr):
        self.loss = loss
        self.lr = lr


    def learn(self,k,x,y,model):
        models = []
        optimizer = optim.SGD(model.parameters(), lr=self.lr)
        sample_num = y.shape[0]
        sep_num = sample_num // k
        for i in range(k):
            for j in range(sep_num):
                ## pytorchでは勾配を蓄積する仕組みなので更新前に初期化しておきます
                optimizer.zero_grad()
                ## feed forward(つまり予測させます)
                output = model(x[i:i+sep_num])
                ## 予実差からの誤差を決めます。nll_lossは、Negative Log Likelihood(負の対数尤度)ですね。
                loss = F.nll_loss(output, y[i:i+sep_num])

                ## Back Propagation
                loss.backward()
            models.append(list(model.parameters())[0])

        return torch.stack(models)

