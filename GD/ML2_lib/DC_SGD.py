import numpy as np
import miniball

from tqdm.notebook import tqdm

from . import algo_sgd
from . import additive_noise


class DCSGD:
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
        # 学習率は固定ではない。
        self.lr = 0.01 / np.sqrt(self.d)
        self.fixed_lr = fixed_lr

    def learn(self, k, w_init):

        model_store = []
        core_store = []

        for _ in range(k):
            core_num = self.n // k
            rng = np.random.default_rng()
            # Xの次元数は(特徴量、サンプル数)
            X = rng.normal(loc=self.X_mean, size=(self.n, self.d), scale=self.X_var)
            tmp = additive_noise.Noise(dim=1, mean=0, sigma=self.E_var, n=self.n)
            # ノイズの次元数は(1,サンプル数)
            E = getattr(tmp, self.noise)()
            # Yの次元数は(1,サンプル数)
            Y = np.dot(self.w_star, X.T) + E
            data = [X, Y.T]
            core = algo_sgd.SGD(w_init=w_init, a=self.lr, t_max=core_num - 1, data=data,fixed_lr=self.fixed_lr)
            for _ in core:
                core.update(self.loss_type)
            core_store.append(core.wstore)
            model_store.append(core.w)

        w_dc, _ = miniball.get_bounding_ball(np.array(model_store).reshape((-1, self.d)), epsilon=1e-7)
        w_dc = w_dc.reshape(1, -1)
        excess_risk = self.loss_type.excess_risk_normal(X_mean=self.X_mean, X_var=self.X_var, w_star=self.w_star,
                                                        w=w_dc)
        return w_dc, excess_risk, core_store

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

        _,_,core_store = self.learn(k=k,w_init=w_init)

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
