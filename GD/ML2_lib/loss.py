import numpy as np

from . import additive_noise

class LinearQuadraticLoss():
    def __init__(self):
        self.type = "loss_with_data"

    def f(self, y, x, w):

        if type(w) == np.float_:
            return np.mean(0.5 * ((y - np.dot(x, w)) ** 2), axis=0)
        elif type(w) == np.ndarray:
            # print("f np.dot(x, w)) {}".format(np.dot(x, w)))
            # print("y {}".format(y))
            # print("f {}".format(np.mean(0.5 * ((y - np.dot(x, w)) ** 2),axis=0)))
            return np.mean(0.5 * ((y - np.dot(x, w)) ** 2), axis=0)
        else:
            raise ValueError('w のデータ型')

    def g(self, y, x, w):
        # print("y:".format(y))
        #
        # print("w * x".format(w * x))
        # print("g  {}".format(- x * (y - np.dot(w, x))))
        return - x * (y - np.dot(w, x))

    def predict(self, x, w):
        return np.dot(w, x)

    def excess_risk_normal(self, X_mean, X_var, w_star, w):
        E_X = np.diag(np.ones(w.shape[0]) * (X_var ** 2) + (X_mean ** 2))
        excess_risk = np.dot(np.dot(E_X, w), w.T) + np.dot(np.dot(E_X, w_star), w_star.T) - 2 * np.dot(
            np.dot(E_X, w_star), w.T)

        return excess_risk[0][0]


class RosenBrock:
    def __init__(self,d,noise_type=None,E_var=1.75):
        self.type = "loss_with_w"
        self.d = d
        self.w_star = np.ones(d)
        self.noise_type = noise_type
        self.E_var = E_var

    def f_opt(self, w):
        w = np.array(w)
        tmp = 0

        for i in range(0, self.d - 1):
            tmp_1 = 100 * (w[i + 1] - w[i] ** 2) ** 2
            tmp_2 = (w[i] - 1) ** 2
            tmp += tmp_1 + tmp_2

        return tmp

    def g_opt(self, w):
        w = np.array(w)
        d = w.shape[0]
        self.w_star = np.ones(d)
        tmp = np.zeros(w.shape)
        for i in range(d):
            if i == 0:
                tmp[i] = 100 * (-4) * w[i] * (w[i + 1] - w[i] ** 2) + 2 * (w[i] - 1)
            elif i == d - 1:
                tmp[i] = 100 * 2 * (w[i] - w[i - 1] ** 2)
            else:
                tmp[i] = 100 * (-4) * w[i] * (w[i + 1] - w[i] ** 2) + 2 * (w[i] - 1) + 100 * 2 * (w[i] - w[i - 1] ** 2)

        if self.noise_type:
            tmp = tmp + self.generate_noise()


        return tmp

    def generate_noise(self):
        tmp = additive_noise.Noise(dim=self.d, mean=0, sigma=self.E_var, n=1)
        E = getattr(tmp, self.noise_type)()

        return E
