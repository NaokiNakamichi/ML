import numpy as np
import miniball
import copy

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
            data = [x[i:i + core_num], y[i:i + core_num]]
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
        self.lr = lr / np.sqrt(self.d)
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

    def multiple_k_transition(self, k_list, w_init):
        k_transition = []
        for k in k_list:
            k_transition.append(self.transition(k=k, w_init=w_init)[1])

        return k_transition


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
    def __init__(self, lr):
        self.lr = lr
        self.model = nn.Module()

    def learn(self, k, x, y, model):
        x = torch.tensor(x).float()
        y = torch.LongTensor(y)
        w_list = []
        bias_list = []
        w_stack = []
        b_stack = []
        sample_num = y.shape[0]
        sep_num = sample_num // k
        for i in range(k):
            model.parameter_init()
            w_transition = []
            b_transition = []
            for j in range(sep_num):
                optimizer = optim.SGD(model.parameters(), lr=self.lr)
                optimizer.zero_grad()
                output = model(x[i:i + sep_num])
                loss = F.nll_loss(output, y[i:i + sep_num])

                # Back Propagation
                loss.backward()
                optimizer.step()

                # 正解率の計算
                prediction = output.data.max(1)[1]
                accuracy = prediction.eq(y[i:i + sep_num]).sum().numpy() / y[i:i + sep_num].shape[0]

                w_transition.append(copy.deepcopy(model.fc1.weight))
                b_transition.append(copy.deepcopy(model.fc1.bias))

            w_stack.append(torch.stack(w_transition))
            b_stack.append(torch.stack(b_transition))
            w_list.append(copy.deepcopy(model.fc1.weight))
            bias_list.append(copy.deepcopy(model.fc1.bias))

        merge_w = merge.median_by_torch(torch.stack(w_list)).values
        merge_b = merge.median_by_torch(torch.stack(bias_list)).values

        with torch.no_grad():
            model.fc1.weight.data = merge_w
            model.fc1.bias.data = merge_b

        # print(w_list)
        # print(merge_w)
        #
        # print(bias_list)
        # print(merge_b)

        self.model = model

        return self.model, w_stack, b_stack

    def predict(self, x, y):
        x = torch.tensor(x).float()
        y = torch.LongTensor(y)
        output = self.model(x)
        prediction = output.data.max(1)[1]
        accuracy = prediction.eq(y).sum().numpy() / y.shape[0]
        print("予測")
        print(prediction)
        print("答え")
        print(y)
        print("正解率")
        print(accuracy)

    def transition(self, k, train_x, train_y, transition_x, transition_y, model):
        # train_x = torch.tensor(train_x).float()
        # train_y = torch.LongTensor(train_y)
        _, w_stack, b_stack = self.learn(k, train_x, train_y, model)
        x = torch.tensor(transition_x).float()
        y = torch.LongTensor(transition_y)
        w_tmp = torch.stack(w_stack).permute(1, 0, 2, 3)
        b_tmp = torch.stack(b_stack).permute(1, 0, 2)
        loss_transition = []
        for w, b in zip(w_tmp, b_tmp):
            merge_w = merge.median_by_torch(w).values
            merge_b = merge.median_by_torch(b).values
            with torch.no_grad():
                model.fc1.weight.data = merge_w
                model.fc1.bias.data = merge_b

            output = model(x)
            loss = F.nll_loss(output, y)
            loss_transition.append(loss.item())

        return loss_transition, model

    def multiple_k_transition(self, k_list, train_x, train_y, transition_x, transition_y, model):
        k_transition = []
        for k in k_list:
            k_transition.append(self.transition(k=k, train_x=train_x, train_y=train_y, transition_x=transition_x,
                                                transition_y=transition_y,
                                                model=model)[0])

        return k_transition

    def multiple_k_accuracy(self, k_list, train_x, train_y, transition_x, transition_y, model):
        k_accuracy = []
        for k in k_list:
            m = self.transition(k=k, train_x=train_x, train_y=train_y, transition_x=transition_x,
                                transition_y=transition_y,
                                model=model)[1]

            x = torch.tensor(transition_x).float()
            y = torch.LongTensor(transition_y)
            output = m(x)
            prediction = output.data.max(1)[1]
            accuracy = prediction.eq(y).sum().numpy() / y.shape[0]
            k_accuracy.append(accuracy)

        return k_accuracy
