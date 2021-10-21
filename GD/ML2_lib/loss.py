import numpy as np

from . import additive_noise


class LossInit():
    def __init__(self, d, noise_type=None, E_var=1.75, noise_type_f=None, f_E_var=1.75):
        self.type = "loss_with_w"
        self.d = d
        self.w_star = np.ones(d)
        self.noise_type = noise_type
        self.E_var = E_var
        self.noise_type_f = noise_type_f
        self.f_E_var = f_E_var

    def generate_noise(self):
        tmp = additive_noise.Noise(dim=self.d, mean=0, sigma=self.E_var, n=1)
        E = getattr(tmp, self.noise_type)()

        return E

    def generate_noise_f(self):
        tmp = additive_noise.Noise(dim=1, mean=0, sigma=self.f_E_var, n=1)
        E = getattr(tmp, self.noise_type)()

        return E

    def remove_f_noise(self):
        self.noise_type_f = None


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


class RosenBrock(LossInit):
    def __init__(self, d, noise_type=None, E_var=1.75, noise_type_f=None, f_E_var=1.75):
        super(RosenBrock, self).__init__(d=d, noise_type=noise_type, E_var=E_var, noise_type_f=noise_type_f,
                                         f_E_var=f_E_var)

    def f_opt(self, w):
        w = np.array(w)
        tmp = 0

        for i in range(0, self.d - 1):
            tmp_1 = 100 * (w[i + 1] - w[i] ** 2) ** 2
            tmp_2 = (w[i] - 1) ** 2
            tmp += tmp_1 + tmp_2

        if self.noise_type_f:
            tmp += self.generate_noise_f()[0]

        return tmp

    def g_opt(self, w):
        w = np.array(w)
        self.w_star = np.ones(self.d)
        tmp = np.zeros(w.shape)
        for i in range(self.d):
            if i == 0:
                tmp[i] = 100 * (-4) * w[i] * (w[i + 1] - w[i] ** 2) + 2 * (w[i] - 1)
            elif i == self.d - 1:
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

    def generate_noise_f(self):
        tmp = additive_noise.Noise(dim=1, mean=0, sigma=self.f_E_var, n=1)
        E = getattr(tmp, self.noise_type)()

        return E

    def remove_f_noise(self):
        self.noise_type_f = None


class Ackley:
    def __init__(self, d, noise_type=None, E_var=1.75, noise_type_f=None, f_E_var=1.75):
        self.type = "loss_with_w"
        self.d = d
        self.w_star = np.zeros(d)
        self.noise_type = noise_type
        self.E_var = E_var
        self.noise_type_f = noise_type_f
        self.f_E_var = f_E_var

    def f_opt(self, w):
        f = 0
        if self.d == 2:
            tmp = np.sqrt(0.5 * (w[0] ** 2 + w[1] ** 2))
            tmp2 = np.cos(2 * np.pi * w[0]) + np.cos(2 * np.pi * w[1])
            f = 20 - 20 * np.exp(-0.2 * tmp) - np.exp(0.5 * tmp2) + np.e

        if self.noise_type_f:
            f += self.generate_noise_f()[0]

        return f

    def g_opt(self, w):
        g = 0
        if self.d == 2:
            tmp = np.sqrt(0.5 * (w[0] ** 2 + w[1] ** 2))
            if type(w) is not int:
                pass

            else:
                if tmp == 0:
                    return np.array([0, 0])
            tmp2 = np.cos(2 * np.pi * w[0]) + np.cos(2 * np.pi * w[1])
            g = (20 * 0.2 * np.exp(- 0.2 * tmp) / tmp) * w + (2 * np.pi * np.sin(np.pi * w) * (np.exp(0.5 * tmp2)))

        if self.noise_type:
            g = g + self.generate_noise()

        return g

    def generate_noise(self):
        tmp = additive_noise.Noise(dim=self.d, mean=0, sigma=self.E_var, n=1)
        E = getattr(tmp, self.noise_type)()

        return E

    def generate_noise_f(self):
        tmp = additive_noise.Noise(dim=1, mean=0, sigma=self.f_E_var, n=1)
        E = getattr(tmp, self.noise_type)()

        return E

    def remove_f_noise(self):
        self.noise_type_f = None


class Bohachevsky:

    def __init__(self, d, noise_type=None, E_var=1.75, noise_type_f=None, f_E_var=1.75):
        self.type = "loss_with_w"
        self.d = d
        self.w_star = np.zeros(d)
        self.noise_type = noise_type
        self.E_var = E_var
        self.noise_type_f = noise_type_f
        self.f_E_var = f_E_var

    def f_opt(self, w):
        w = np.array(w)
        w1, w2 = w[0], w[1]
        f = w1 ** 2 + 2 * w2 ** 2 - 0.3 * np.cos(3 * np.pi * w1) - 0.4 * np.cos(4 * np.pi * w2) + 0.7
        return f

    def g_opt(self, w):
        w = np.array(w)
        w1, w2 = w[0], w[1]
        g_w1 = 2 * w1 + 0.9 * np.pi * np.sin(3 * np.pi * w1)
        g_w2 = 4 * w2 + 1.6 * np.pi * np.sin(4 * np.pi * w2)
        g = np.array([g_w1, g_w2])
        g = g + self.noise_value
        return g
