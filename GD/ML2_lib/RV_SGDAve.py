import numpy as np
from tqdm.notebook import tqdm

from . import loss
from . import algo_sgd
from . import additive_noise
from . import valid


class RVSGD:
    def __init__(self, w_star, n, E_var, X_mean, X_var, noise, loss_type, c, fixed_lr=True):
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
            self.loss_type.type
        except:
            raise ValueError("損失を確認してください。")
        self.c = c
        self.d = self.w_star.shape[1]
        self.lr = 0.01 / np.sqrt(self.d)
        # 学習率は固定かどうか選択する。
        self.fixed_lr = fixed_lr

    def learn(self, k, w_init):
        core_store = []
        model_store = []
        valid_loss_store = []

        son = loss.LinearQuadraticLoss()
        for _ in range(k):
            # nがデータセットのサンプル数、train_numはその半分
            train_num = self.n // 2
            # core_num は　k分割した後のサンプル数、
            core_num = train_num // k
            rng = np.random.default_rng()
            X = rng.normal(loc=self.X_mean, size=(self.n, self.d), scale=self.X_var)
            tmp = additive_noise.Noise(dim=self.d, mean=0, sigma=self.E_var, n=self.n)
            E = getattr(tmp, self.noise)()
            Y = np.dot(self.w_star, X.T) + E
            data = [X, Y.T]

            core = algo_sgd.SGD(w_init=w_init, a=self.lr, t_max=core_num - 1, data=data)
            for _ in core:
                core.update(son)
            core_store.append(core)
            # :TODO fix axis
            model_store.append(np.mean(core.wstore))

        # ここまでで学習は終了,モデルの候補がk個ある
        # ここからモデルの選択
        valid_num = self.n // 2
        rng = np.random.default_rng()
        X = rng.normal(loc=self.X_mean, size=(self.n, self.d), scale=self.X_var)
        tmp = additive_noise.Noise(dim=self.d, mean=0, sigma=self.E_var, n=self.n)
        E = getattr(tmp, self.noise)()
        Y = np.dot(self.w_star, X.T) + E
        tmp_loss = []

        # for文を使っているので要修正
        for i in range(k):
            for j in range(k):
                core_num = valid_num // k
                try:
                    tmp_loss.append(son.f(Y[j:j + core_num], X[j:j + core_num], model_store[i]))
                except:
                    raise ValueError()
            valid_loss_store.append(valid.median_of_means(seq=np.array(tmp_loss), n_blocks=3))

        index = np.argmin(valid_loss_store)
        w_rv = model_store[index]
        # 過剰期待損失　E[(<(w-w^*),X>)^2]　Xが正規分布の場合　E[X^2] = X_mean^2 + X_var^2 * 単位行列
        E_X = np.diag(np.ones(self.w_star.shape[0]) * (self.X_var ** 2) + (self.X_mean ** 2))
        excess_risk = np.dot(np.dot(E_X, w_rv), w_rv) + np.dot(np.dot(E_X, self.w_star), self.w_star) - 2 * np.dot(
            np.dot(E_X, self.w_star), w_rv)

        return w_rv, excess_risk
