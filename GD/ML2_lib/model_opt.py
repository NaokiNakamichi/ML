import numpy as np

from . import models

"""
This nodule give test function value and the gradient.
It can add only just additive noise. it can't add multiplicative noise.
w_star is value that minimize test function value.

input is numpy.array or array
f_value output is scalar, gradient output depends on input dimension
"""



class Bohachevsky(models.Model):
    def __init__(self, name="Bohachevsky", noise_value=np.array([0, 0])):
        super(Bohachevsky, self).__init__(name=name)
        self.w_star = np.array([0, 0])
        self.noise_value = noise_value

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


class Perm(models.Model):
    def __init__(self, name="Perm", b=0.001, noise_value=np.array([0, 0])):
        super(Perm, self).__init__(name=name)
        self.b = b
        self.noise_value = noise_value
        self.w_star = []
        self.d = self.noise_value.shape[-1]
        for i in range(self.d):
            self.w_star.append(1 / (i + 1))
        self.w_star = np.array(self.w_star)

    def f_opt(self, w):
        w = np.array(w)
        tmp_1 = np.arange(1, self.d + 1)
        tmp_2 = 1 / tmp_1
        tmp_3 = tmp_1 + self.b
        tmp = 0
        for i in range(1, self.d + 1):
            tmp += np.sum(tmp_3 * (w ** i - tmp_2 ** i)) ** 2

        return tmp

    def g_opt(self, w):
        w = np.array(w)
        tmp = np.zeros(w.shape)
        tmp_1 = np.arange(1, self.d + 1)
        tmp_2 = 1 / tmp_1
        tmp_3 = tmp_1 + self.b
        # TODO:for文じゃないやり方で。連番配列用意してかけるとか
        for i in range(1, self.d + 1):
            tmp += 2 * i * w ** (i - 1) * tmp_3 ** 2 * (w ** i - tmp_2 ** i)

        tmp += self.noise_value

        return tmp


class RotatedHyperEllipsoid(models.Model):
    def __init__(self, name="ROTATED HYPER-ELLIPSOID", noise_value=np.array([0, 0])):
        super(RotatedHyperEllipsoid, self).__init__(name=name)
        self.w_star = np.array([0, 0])
        self.noise_value = noise_value
        self.d = self.noise_value.shape[-1]

    def f_opt(self, w):
        w = np.array(w)
        tmp1 = np.arange(self.d, 0, -1)
        tmp2 = w ** 2
        tmp = np.sum(tmp1 * tmp2)
        return tmp

    def g_opt(self, w):
        w = np.array(w)
        tmp1 = np.arange(self.d, 0, -1)
        tmp = 2 * tmp1 * w
        tmp = tmp + self.noise_value
        return tmp


class Sphere(models.Model):
    def __init__(self, name="Sphere", noise_value=np.array([0, 0])):
        super(Sphere, self).__init__(name=name)
        self.w_star = np.array([0, 0])
        self.noise_value = noise_value
        self.d = self.noise_value.shape[-1]

    def f_opt(self, w):
        w = np.array(w)
        tmp = np.sum(w ** 2)
        return tmp

    def g_opt(self, w):
        w = np.array(w)
        tmp = 2 * w
        tmp = tmp + self.noise_value
        return tmp


class SumOfDifferent(models.Model):
    def __init__(self, name="SumOfDifferent", noise_value=np.array([0, 0])):
        super(SumOfDifferent, self).__init__(name=name)
        self.w_star = np.array([0, 0])
        self.noise_value = noise_value
        self.d = self.noise_value.shape[-1]

    def f_opt(self, w):
        w = np.array(w)
        tmp1 = np.arange(2, self.d + 2)
        tmp = np.sum(np.abs(w) ** tmp1)
        return tmp

    def g_opt(self, w):
        w = np.array(w)
        tmp1 = np.arange(2, self.d + 2)
        tmp2 = []
        for i, j in enumerate(w):
            if (j < 0) and (tmp1[i] % 2 == 1):
                tmp2.append(-1)
            else:
                tmp2.append(1)

        tmp = tmp1 * np.array(tmp2) * w ** (tmp1 - 1)
        tmp = tmp + self.noise_value

        return tmp


class SumSquares(models.Model):
    def __init__(self, name="Sum Squares", noise_value=np.array([0, 0])):
        super(SumSquares, self).__init__(name=name)
        self.w_star = np.array([0, 0])
        self.noise_value = noise_value
        self.d = self.noise_value.shape[-1]

    def f_opt(self, w):
        w = np.array(w)
        tmp1 = np.arange(1, self.d + 1)
        tmp = np.sum(tmp1 * (w ** 2))
        return tmp

    def g_opt(self, w):
        w = np.array(w)
        tmp1 = np.arange(1, self.d + 1)
        tmp = 2 * tmp1 * w
        tmp = tmp + self.noise_value
        return tmp


class Trid(models.Model):
    def __init__(self, name="trid", noise_value=np.array([0, 0])):
        super(Trid, self).__init__(name=name)
        self.w_star = []
        self.noise_value = noise_value
        self.d = self.noise_value.shape[-1]
        for i in range(self.d):
            minimum = (i + 1) * (self.d + 1 - (i + 1))
            self.w_star.append(minimum)
        self.w_star = np.array(self.w_star)

    def f_opt(self, w):
        w = np.array(w)
        tmp1 = np.sum(w[1:] * w[:-1])
        tmp2 = np.sum((w - 1) ** 2)
        tmp = tmp2 - tmp1

        return tmp

    def g_opt(self, w):
        w = np.array(w)
        tmp = np.zeros(w.shape)
        for i in range(self.d):
            if i == 0:
                tmp[i] = 2 * (w[i] - 1) - w[i + 1]
            elif i == self.d - 1:
                tmp[i] = 2 * (w[i] - 1) - w[i - 1]
            else:
                tmp[i] = 2 * (w[i] - 1) - (w[i - 1] + w[i + 1])

        tmp = tmp + self.noise_value

        return tmp


class ThreeHumpCamel(models.Model):
    def __init__(self, name="Three-Hump Camel", err=0.0):
        super(ThreeHumpCamel, self).__init__(name=name)
        self.err = err
        self.w_star = np.array([0, 0])

    def f_opt(self, w):
        w1, w2 = w[0], w[1]
        tmp = 2 * w1 ** 2 - 1.05 * w1 ** 4 + (w1 ** 6) / 6 + w1 * w2 + w2 ** 2
        if self.err != 0:
            tmp = tmp + self.err * np.random.randn(1)
        return tmp

    def g_opt(self, w):
        w1, w2 = w[0], w[1]
        tmp_1 = 4 * w1 - 4.2 * w1 ** 3 + w1 ** 5 + w2
        tmp_2 = w1 + 2 * w2
        tmp = np.array([tmp_1, tmp_2])
        if self.err != 0:
            tmp = tmp + self.err * np.random.randn(2)
        return tmp


class SixHumpCamel(models.Model):
    def __init__(self, name="Six-Hump Camel", err=0.0):
        super(SixHumpCamel, self).__init__(name=name)
        self.err = err
        self.w_star = np.array([0.0898, -0.7126])

    def f_opt(self, w):
        w1, w2 = w[0], w[1]
        tmp = (4 - 2.1 * (w1 ** 2) + (w1 ** 4) / 3) * (w1 ** 2) + w1 * w2 + (-4 + 4 * (w2 ** 2)) * (w2 ** 2)
        if self.err != 0:
            tmp = tmp + self.err * np.random.randn(1)
        return tmp

    def g_opt(self, w):
        w1, w2 = w[0], w[1]
        tmp_1 = 8 * w1 - 8.4 * (w1 ** 3) + 2 * (w1 ** 5) + w2
        tmp_2 = w1 - 8 * w2 + 16 * (w2 ** 3)
        tmp = np.array([tmp_1, tmp_2])
        if self.err != 0:
            tmp = tmp + self.err * np.random.randn(1)
        return tmp


class DixonPrice(models.Model):
    def __init__(self, name="Dixon-Price", noise_value=np.array([0, 0])):
        super(DixonPrice, self).__init__(name=name)
        self.w_star = np.array([1, 2 ** (-1 / 2)])
        self.noise_value = noise_value

    def f_opt(self, w):
        w = np.array(w)
        d = w.shape[0]
        self.w_star = []
        for i in range(1, d + 1):
            minimum = 2 ** ((-1) * ((2 ** i - 2) / 2 ** i))
            self.w_star.append(minimum)
        self.w_star = np.array(self.w_star)
        tmp = np.zeros(w.shape)
        for i in range(1, d):
            tmp += (i + 1) * (2 * w[i] ** 2 - w[i - 1]) ** 2
        tmp += (w[0] - 1) ** 2
        return tmp[0]

    def g_opt(self, w):
        w = np.array(w)
        d = w.shape[0]
        self.w_star = []
        for i in range(1, d + 1):
            minimum = 2 ** ((-1) * ((2 ** i - 2) / 2 ** i))
            self.w_star.append(minimum)
        self.w_star = np.array(self.w_star)
        tmp = np.zeros(w.shape)
        for i in range(d):
            if i == 0:
                tmp[i] = 2 * (w[i] - 1) - 2 * 2 * (2 * w[i + 1] ** 2 - w[i])
            elif i == d - 1:
                tmp[i] = 8 * w[i] * d * (i + 1) * (2 * w[i] ** 2 - w[i - 1])
            else:
                tmp[i] = 8 * w[i] * d * (i + 1) * (2 * w[i] ** 2 - w[i - 1]) - (i + 2) * 2 * (2 * w[i + 1] ** 2 - w[i])

        tmp = tmp + self.noise_value

        return tmp


class RosenBrock(models.Model):
    def __init__(self, name="RosenBrock", noise_value=np.array([0, 0])):
        super(RosenBrock, self).__init__(name=name)
        self.w_star = np.ones(2)
        self.noise_value = noise_value

    def f_opt(self, w):
        w = np.array(w)
        d = w.shape[0]
        tmp = 0
        self.w_star = np.ones(d)
        for i in range(0, d - 1):
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

        tmp = tmp + self.noise_value

        return tmp
