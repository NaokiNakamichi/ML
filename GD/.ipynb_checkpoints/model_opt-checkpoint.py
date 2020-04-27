import numpy as np

import models
import random

class Bohachevsky(models.Model):
    # minimum (0,0)

    def __init__(self, name="Bohachevsky",err=0.0):
        super(Bohachevsky, self).__init__(name=name)
        self.err = err
        self.w_star = np.array([0,0])

    def f_opt(self,w):
        w =  np.array(w)
        w1,w2 = w[0],w[1]
        f = w1 ** 2 + 2 * w2 ** 2 - 0.3 * np.cos(3 * np.pi * w1) - 0.4 * np.cos(4* np.pi * w2) + 0.7
        dim = f.shape
        if self.err != 0:
            f = f + self.err * np.random.randn(1)
        return f

    def g_opt(self,w):
        w =  np.array(w)
        w1,w2 = w[0],w[1]
        g_w1 = 2 * w1 + 0.9 * np.pi * np.sin(3 * np.pi * w1)
        g_w2 = 4 * w2 + 1.6 * np.pi * np.sin(4 * np.pi * w2)
        g = np.array([g_w1,g_w2])
        if self.err != 0:
            g = g + self.err * np.random.randn(2)
        return g




# TODO: Perm関数　SGDがうまく収束していない、勾配がうまく実装できてない可能性。
class Perm(models.Model):
    def __init__(self, name="Perm",err=0.0,b=0.001):
        super(Perm, self).__init__(name=name)
        self.w_star = np.array([1,0.5])
        self.err = err
        self.b = b

    def f_opt(self,w):
        w = np.array(w)
        d = w.shape[0]
        self.w_star = []
        for i in range(d):
            self.w_star.append(1 / (i + 1))
        self.w_star = np.array(self.w_star)

        tmp = 0
        for i in list(range(d)):
            for j in list(range(d)):
                tmp += ((j + 1 + self.b) * (w[j] ** (i + 1) - (1 / (j + 1) ** (i + 1)))) ** 2

        if self.err != 0:
            tmp = tmp + self.err * np.random.randn(1)

        return tmp

    def g_opt(self,w):
        w = np.array(w)
        d = w.shape[0]
        self.w_star = []
        for i in range(d):
            self.w_star.append(1 / (i + 1))
        self.w_star = np.array(self.w_star)
        tmp = np.zeros(w.shape)
        for i in list(range(d)):
            for j in list(range(d)):
                for k in list(range(d)):
                    s = (j + 1 + self.b) * (2 * (i+1) * (w[j] ** (i))) * (w[k] ** (i + 1) - (1 / (j + 1) ** (i + 1)))
                    tmp[j] += s

        if self.err != 0:
            tmp = tmp + self.err * np.random.randn(d)
        return tmp

class RotatedHyperEllipsoid(models.Model):
    def __init__(self, name="ROTATED HYPER-ELLIPSOID",err=0.0):
        super(RotatedHyperEllipsoid, self).__init__(name=name)
        self.err = err
        self.w_star = np.array([0,0])

    def f_opt(self, w):
        w = np.array(w)
        d = w.shape[0]
        self.w_star = np.zeros(d)
        tmp = np.zeros(w.shape)
        for i in range(d):
            tmp += np.sum(w[:i + 1] ** 2,axis=0)
        if self.err != 0:
            tmp = tmp + self.err * np.random.randn(1)
        return tmp[0]

    def g_opt(self,w):
        w = np.array(w)
        d = w.shape[0]
        self.w_star = np.zeros(d)
        tmp = np.zeros(w.shape)
        for i in range(d):
            tmp[i] += np.sum(2 * w[:i + 1],axis=0)

        if self.err != 0:
            tmp = tmp + self.err * np.random.randn(1)
        return tmp

class Sphere(models.Model):
    def __init__(self, name="Sphere",err=0.0):
        super(Sphere, self).__init__(name=name)
        self.err = err
        self.w_star = np.array([0,0])

    def f_opt(self, w):
        w = np.array(w)
        d = w.shape[0]
        self.w_star = np.zeros(d)
        tmp = np.zeros(w.shape)
        for i in range(d):
            tmp += w[i] ** 2

        if self.err != 0:
            tmp = tmp + self.err * np.random.randn(1)

        return tmp[0]

    def g_opt(self,w):
        w = np.array(w)
        d = w.shape[0]
        self.w_star = np.zeros(d)
        tmp = np.zeros(w.shape)
        for i in range(d):
            tmp[i] = 2 * w[i]

        if self.err != 0:
            tmp = tmp + self.err * np.random.randn(1)

        return tmp

class SumOfDifferent(models.Model):
    def __init__(self, name="SumOfDifferent",err=0.0):
        super(SumOfDifferent, self).__init__(name=name)
        self.err = err
        self.w_star = np.array([0,0])

    def f_opt(self,w):
        w = np.array(w)
        d = w.shape[0]
        self.w_star = np.zeros(d)
        tmp = np.zeros(w.shape)
        for i in range(d):
            tmp += np.abs(w[i]) ** (i + 2)

        if self.err != 0:
            tmp = tmp + self.err * np.random.randn(1)

        return tmp[0]

    def g_opt(self,w):
        w = np.array(w)
        d = w.shape[0]
        self.w_star = np.zeros(d)
        tmp = np.zeros(w.shape)
        for i in range(d):
            tmp[i] =  (i + 2) * (np.abs(w[i]) ** (i + 1))

        if self.err != 0:
            tmp = tmp + self.err * np.random.randn(1)

        return tmp

class SumSquares(models.Model):
    def __init__(self, name="Sum Squares",err=0.0):
        super(SumSquares, self).__init__(name=name)
        self.err = err
        self.w_star = np.array([0,0])

    def f_opt(self,w):
        w = np.array(w)
        d = w.shape[0]
        self.w_star = np.zeros(d)
        tmp = np.zeros(w.shape)
        for i in range(d):
            tmp += (i + 1) * (w[i] ** 2)

        if self.err != 0:
            tmp = tmp + self.err * np.random.randn(1)

        return tmp[0]

    def g_opt(self,w):
        w = np.array(w)
        d = w.shape[0]
        self.w_star = np.zeros(d)
        tmp = np.zeros(w.shape)
        for i in range(d):
            tmp[i] = 2 * (i + 1) * (w[i])

        if self.err != 0:
            tmp = tmp + self.err * np.random.randn(1)

        return tmp

class Trid(models.Model):
    def __init__(self, name="trid",err=0.0):
        super(Trid, self).__init__(name=name)
        self.err = err
        self.w_star = [2,2]

    def f_opt(self,w):
        w = np.array(w)
        d = w.shape[0]
        self.w_star = []
        for i in range(d):
            minimum = (i + 1) * (d + 1 - (i + 1))
            self.w_star.append(minimum)
        self.w_star = np.array(self.w_star)

        tmp_1 = np.zeros(w.shape)
        tmp_2 = np.zeros(w.shape)
        for i in range(d):
            tmp_1 += (w[i] - 1) ** 2
        for i in range(1,d):
            tmp_2 += w[i] * w[i-1]
        tmp = tmp_1 - tmp_2

        if self.err != 0:
            tmp = tmp + self.err * np.random.randn(1)

        return tmp[0]

    def g_opt(self,w):
        w = np.array(w)
        d = w.shape[0]
        self.w_star = []
        for i in range(d):
            minimum = (i + 1) * (d + 1 - (i + 1))
            self.w_star.append(minimum)
        self.w_star = np.array(self.w_star)
        tmp = np.zeros(w.shape)
        for i in range(d):
            if i == 0:
                tmp[i] = 2 * (w[i] - 1) - w[i+1]
            elif i == d-1:
                tmp[i] = 2 * (w[i] - 1) - w[i-1]
            else:
                tmp[i] = 2 * (w[i] - 1) - (w[i-1] + w[i+1])

        if self.err != 0:
            tmp = tmp + self.err * np.random.randn(1)

        return tmp

class ThreeHumpCamel(models.Model):
    def __init__(self, name="Three-Hump Camel",err=0.0):
        super(ThreeHumpCamel, self).__init__(name=name)
        self.err = err
        self.w_star = np.array([0,0])

    def f_opt(self,w):
        w1,w2 = w[0],w[1]
        tmp = 2 * w1 ** 2 - 1.05 * w1 ** 4 + (w1 ** 6) / 6 + w1 * w2 + w2 ** 2
        if self.err != 0:
            tmp = tmp + self.err * np.random.randn(1)
        return tmp

    def g_opt(self,w):
        w1,w2 = w[0],w[1]
        tmp_1 = 4 * w1 - 4.2 * w1 ** 3 + w1 ** 5 + w2
        tmp_2 = w1 + 2 * w2
        tmp = np.array([tmp_1,tmp_2])
        if self.err != 0:
            tmp = tmp + self.err * np.random.randn(2)
        return tmp

class SixHumpCamel(models.Model):
    def __init__(self, name="Six-Hump Camel",err=0.0):
        super(SixHumpCamel, self).__init__(name=name)
        self.err = err
        self.w_star = np.array([0.0898,-0.7126])

    def f_opt(self,w):
        w1,w2 = w[0],w[1]
        tmp = (4 - 2.1 * (w1 ** 2) + (w1 ** 4) / 3 ) * (w1 ** 2) + w1 * w2 + (-4 + 4 * (w2 ** 2)) * (w2 ** 2)
        if self.err != 0:
            tmp = tmp + self.err * np.random.randn(1)
        return tmp

    def g_opt(self,w):
        w1,w2 = w[0],w[1]
        tmp_1 = 8 * w1 - 8.4 * (w1 ** 3) + 2 * (w1 ** 5) + w2
        tmp_2 = w1 - 8 * w2 + 16 * (w2 ** 3)
        tmp = np.array([tmp_1,tmp_2])
        if self.err != 0:
            tmp = tmp + self.err * np.random.randn(1)
        return tmp

class DixonPrice(models.Model):
    def __init__(self, name="Dixon-Price",err=0.0):
        super(DixonPrice, self).__init__(name=name)
        self.err = err
        self.w_star = np.array([1, 2 ** (-1/2)])

    def f_opt(self,w):
        w = np.array(w)
        d = w.shape[0]
        self.w_star = []
        for i in range(1,d+1):
            minimum = 2 ** ((-1) * ((2 ** i - 2) / 2 ** i))
            self.w_star.append(minimum)
        self.w_star = np.array(self.w_star)
        tmp = np.zeros(w.shape)
        for i in range(1,d):
            tmp += (i+1) * (2 * w[i] ** 2  - w[i-1]) ** 2
        tmp += (w[0]-1) ** 2

        if self.err != 0:
            tmp = tmp + self.err * np.random.randn(1)

        return tmp[0]

    def g_opt(self,w):
        w = np.array(w)
        d = w.shape[0]
        self.w_star = []
        for i in range(1,d+1):
            minimum = 2 ** ((-1) * ((2 ** i - 2) / 2 ** i))
            self.w_star.append(minimum)
        self.w_star = np.array(self.w_star)
        tmp = np.zeros(w.shape)
        for i in range(d):
            if i == 0:
                tmp[i] = 2 * (w[i] - 1) - 2 * 2 * (2 * w[i+1] ** 2 - w[i])
            elif i == d-1:
                tmp[i] = 8 * w[i] * d * (i + 1) * (2 * w[i] ** 2 - w[i-1])
            else:
                tmp[i] =  8 * w[i] * d * (i + 1) * (2 * w[i] ** 2 - w[i-1]) - (i + 2) * 2 * (2 * w[i+1] ** 2 - w[i])

        if self.err != 0:
            tmp = tmp + self.err * np.random.randn(1)

        return tmp


class RosenBrock(models.Model):
    def __init__(self, name="RosenBrock",err=0.0):
        super(RosenBrock, self).__init__(name=name)
        self.err = err
        self.w_star = np.ones(2)

    def f_opt(self,w):
        w = np.array(w)
        d = w.shape[-1]
        self.w_star = np.ones(d)
        tmp = np.zeros(w.shape[0])
        for j in range(tmp.shape[0]):
            for i in range(0, d - 1):
                tmp_1 = 100 * (w[j][i+1] - w[j][i] ** 2) ** 2
                tmp_2 = (w[j][i] -1) ** 2
                tmp[j] += tmp_1 + tmp_2

        tmp = tmp + self.err * (np.random.random_sample(tmp.shape) * random.random())

        return tmp

    def g_opt(self,w):
        w = np.array(w)
        d = w.shape[0]
        self.w_star = np.ones(d)
        tmp = np.zeros(w.shape)
        for i in range(d):
            if i == 0:
                tmp[i] = 100 * (-4) * w[i] * (w[i+1] - w[i] ** 2) + 2 * (w[i] - 1)
            elif i == d-1:
                tmp[i] = 100 * 2 * (w[i] - w[i-1] ** 2)
            else:
                tmp[i] = 100 * (-4) * w[i] * (w[i+1] - w[i] ** 2) + 2 * (w[i] - 1) + 100 * 2 * (w[i] - w[i-1] ** 2)

        tmp = self.add_noise(tmp)

        return tmp

class Quadratic(models.Model):
    def __init__(self,name="Quadratic",err=0.0):
        super(Quadratic, self).__init__(name=name)

    def f_opt(self,w):
        return w**2

    def g_opt(self,w):
        return 2 * w
