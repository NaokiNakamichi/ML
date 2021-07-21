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

        for _ in range(k):
            # nがデータセットのサンプル数、train_numはその半分
            train_num = self.n // 2
            # core_num は　k分割した後のサンプル数、
            core_num = train_num // k
            rng = np.random.default_rng()
            X = rng.normal(loc=self.X_mean, size=(self.n, self.d), scale=self.X_var)
            tmp = additive_noise.Noise(dim=1, mean=0, sigma=self.E_var, n=self.n)
            E = getattr(tmp, self.noise)()
            Y = np.dot(self.w_star, X.T) + E
            data = [X, Y.T]

            core = algo_sgd.SGD(w_init=w_init, a=self.lr, t_max=core_num - 1, data=data)
            for _ in core:
                core.update(self.loss_type)
            core_store.append(core.wstore)
            model_store.append(np.mean(core.wstore, axis=0))

        # ここまでで学習は終了,モデルの候補がk個ある
        # ここからモデルの選択
        valid_num = self.n // 2
        rng = np.random.default_rng()
        X = rng.normal(loc=self.X_mean, size=(self.n, self.d), scale=self.X_var)
        tmp = additive_noise.Noise(dim=1, mean=0, sigma=self.E_var, n=self.n)
        E = getattr(tmp, self.noise)()
        Y = (np.dot(self.w_star, X.T) + E).T
        tmp_loss = []

        # for文を使っているので要修正
        for i in range(k):
            for j in range(k):
                core_num = valid_num // k
                try:
                    tmp_loss.append(
                        self.loss_type.f(Y[j:j + core_num, :], X[j:j + core_num, :], model_store[i].T).shape)
                except:
                    raise ValueError("なんか入力値がおかしい気がする")
            valid_loss_store.append(valid.median_of_means(seq=np.array(tmp_loss), n_blocks=3))

        index = np.argmin(valid_loss_store)
        w_rv = model_store[index]
        excess_risk = self.loss_type.excess_risk_normal(X_mean=self.X_mean, X_var=self.X_var, w_star=self.w_star,
                                                        w=w_rv)

        return w_rv, excess_risk, core_store

    def trial_k(self, max_k):
        w_trial = []  # モデルを貯めていく、必要かどうか
        loss_store = []  # 過剰期待損失を貯めていく
        if max_k < 1:
            raise ValueError("max_k は0以上の整数")
        rng = np.random.default_rng()
        w_init = self.w_star + rng.uniform(-self.c, self.c, size=self.d)
        for k in range(1, max_k + 1):
            w, excess_risk, _ = self.learn(k=k, w_init=w_init)
            w_trial.append(w)
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

        _, _, core_store = self.learn(k=k, w_init=w_init)

        w_transition = []
        loss_transition = []

        core_store = np.array(core_store)

        tmp = core_store.shape

        core_num, update_num, _, w_dim = tmp
        core_store = core_store.reshape(core_num, update_num, w_dim)
        core_store = core_store.transpose(1, 0, 2)

        valid_num = self.n // 2
        rng = np.random.default_rng()
        X = rng.normal(loc=self.X_mean, size=(self.n, self.d), scale=self.X_var)
        tmp = additive_noise.Noise(dim=1, mean=0, sigma=self.E_var, n=self.n)
        E = getattr(tmp, self.noise)()
        Y = (np.dot(self.w_star, X.T) + E).T
        tmp_loss = []

        for i in range(2,update_num):
            w = np.mean(core_store[:i,:, :],axis=0)
            valid_loss_store = []

            for i in range(k):
                for j in range(k):
                    core_num = valid_num // k
                    try:
                        tmp_loss.append(
                            self.loss_type.f(Y[j:j + core_num, :], X[j:j + core_num, :], w[i].T))
                    except:
                        raise ValueError("なんか入力値がおかしい気がする")
                valid_loss_store.append(valid.median_of_means(seq=np.array(tmp_loss), n_blocks=3))

            index = np.argmin(valid_loss_store)
            w_rv = w[index].reshape(1,self.d)
            excess_risk = self.loss_type.excess_risk_normal(X_mean=self.X_mean, X_var=self.X_var, w_star=self.w_star,
                                                            w=w_rv)
            loss_transition.append(excess_risk)
        return loss_transition
