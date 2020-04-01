import numpy as np

import models

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
        if dim:
            return  f + self.err * np.random.randn(dim[0],dim[1])
        else:
            return  f + self.err * np.random.randn(1)

    def g_opt(self,w):
        w =  np.array(w)
        if w.ndim == 1:
            w1,w2 = w[0],w[1]
            g_w1 = 2 * w1 + 0.9 * np.pi * np.sin(3 * np.pi * w1)
            g_w2 = 4 * w2 + 1.6 * np.pi * np.sin(4 * np.pi * w2)
            g = np.array([g_w1,g_w2])
            return g
        else:
            w = w.T
            w1,w2 = w[0],w[1]
            g_w1 = 2 * w1 + 0.9 * np.pi * np.sin(3 * np.pi * w1)
            g_w1 = np.mean(g_w1)
            g_w2 = 4 * w2 + 1.6 * np.pi * np.sin(4 * np.pi * w2)
            g_w2 = np.mean(g_w2)
            g = np.array([g_w1,g_w2])
            return g




class Perm(models.Model):
    def __init__(self, name="Perm",err=0.0):
        super(Perm, self).__init__(name=name)
        self.err = err

    def f_opt(self,w,b):
        w = np.array(w)
        d = w.shape[0]
        tmp = 0
        for i in list(range(d)):
            for j in list(range(d)):
                tmp += ((j + 1 + b) * (w[j] ** (i + 1) - (1 / (j + 1) ** (i + 1)))) ** 2

        return tmp

    def g_opt(self,w,b):
        w = np.array(w)
        d = w.shape[0]

        print(w)
        print(d)
        tmp = np.zeros(w.shape)
        print(tmp)
        for i in list(range(d)):
            for j in list(range(d)):
                for k in list(range(d)):
                    s = (j + 1 + b) * (2 * (w[j] ** i)) * (w[k] ** (i + 1) - (1 / (j + 1) ** (i + 1)))
                    tmp[j] += s

        return tmp

class RotatedHyperEllipsoid(models.Model):
    def __init__(self, name="ROTATED HYPER-ELLIPSOID",err=0.0):
        super(RotatedHyperEllipsoid, self).__init__(name=name)
        self.err = err

    def f_opt(self, w):
        w = np.array(w)
        d = w.shape[0]
        tmp = np.zeros(w.shape)
        for i in range(d):
            tmp += np.sum(w[:i + 1] ** 2,axis=0)
        return tmp[0]

    def g_opt(self,w):
        w = np.array(w)
        d = w.shape[0]
        tmp = np.zeros(w.shape)
        for i in range(d):
            tmp[i] += np.sum(2 * w[:i + 1],axis=0)
        return tmp

class Sphere(models.Model):
    def __init__(self, name="Sphere",err=0.0):
        super(Sphere, self).__init__(name=name)
        self.err = err

    def f_opt(self, w):
        w = np.array(w)
        d = w.shape[0]
        tmp = np.zeros(w.shape)
        for i in range(d):
            tmp += w[i] ** 2

        return tmp[0]

    def g_opt(self,w):
        w = np.array(w)
        d = w.shape[0]
        tmp = np.zeros(w.shape)
        for i in range(d):
            tmp[i] = 2 * w[i]

        return tmp

class SumOfDifferent(models.Model):
    def __init__(self, name="SumOfDifferent",err=0.0):
        super(SumOfDifferent, self).__init__(name=name)
        self.err = err

    def f_opt(self,w):
        w = np.array(w)
        d = w.shape[0]
        tmp = np.zeros(w.shape)
        for i in range(d):
            tmp += np.abs(w[i]) ** (i + 2)

        return tmp[0]

    def g_opt(self,w):
        w = np.array(w)
        d = w.shape[0]
        tmp = np.zeros(w.shape)
        for i in range(d):
            tmp[i] =  (i + 2) * (np.abs(w[i]) ** (i + 1))

        return tmp

class SumSquares(models.Model):
    def __init__(self, name="Sum Squares",err=0.0):
        super(SumSquares, self).__init__(name=name)
        self.err = err

    def f_opt(self,w):
        w = np.array(w)
        d = w.shape[0]
        tmp = np.zeros(w.shape)
        for i in range(d):
            tmp += (i + 1) * (w[i] ** 2)

        return tmp[0]

    def g_opt(self,w):
        w = np.array(w)
        d = w.shape[0]
        tmp = np.zeros(w.shape)
        for i in range(d):
            tmp[i] = 2 * (i + 1) * (w[i])

        return tmp

class Trid(models.Model):
    def __init__(self, name="trid",err=0.0):
        super(Trid, self).__init__(name=name)
        self.err = err

    def f_opt(self,w):
        w = np.array(w)
        d = w.shape[0]
        tmp_1 = np.zeros(w.shape)
        tmp_2 = np.zeros(w.shape)
        for i in range(d):
            tmp_1 += (w[i] - 1) ** 2
        for i in range(1,d):
            tmp_2 += w[i] * w[i-1]
        tmp = tmp_1 - tmp_2

        return tmp[0]

    def g_opt(self,w):
        w = np.array(w)
        d = w.shape[0]
        tmp = np.zeros(w.shape)
        for i in range(d):
            if i == 0:
                tmp[i] = 2 * (w[i] - 1) - w[i+1]
            elif i == d-1:
                tmp[i] = 2 * (w[i] - 1) - w[i-1]
            else:
                tmp[i] = 2 * (w[i] - 1) - (w[i-1] + w[i+1])

        return tmp

class ThreeHumpCamel(models.Model):
    def __init__(self, name="Three-Hump Camel",err=0.0):
        super(ThreeHumpCamel, self).__init__(name=name)
        self.err = err

    def f_opt(self,w):
        w1,w2 = w[0],w[1]
        return 2 * w1 ** 2 - 1.05 * w1 ** 4 + (w1 ** 6) / 6 + w1 * w2 + w2 ** 2

    def g_opt(self,w):
        w1,w2 = w[0],w[1]
        tmp_1 = 4 * w1 - 4.2 * w1 ** 3 + w1 ** 5 + w2
        tmp_2 = w1 + 2 * w2
        return [tmp_1,tmp_2]

class SixHumpCamel(models.Model):
    def __init__(self, name="Six-Hump Camel",err=0.0):
        super(SixHumpCamel, self).__init__(name=name)
        self.err = err

    def f_opt(self,w):
        w1,w2 = w[0],w[1]
        return (4 - 2.1 * (w1 ** 2) + (w1 ** 4) / 3 ) * (w1 ** 2) + w1 * w2 + (-4 + 4 * (w2 ** 2)) * (w2 ** 2)

    def g_opt(self,w):
        w1,w2 = w[0],w[1]
        tmp_1 = 8 * w1 - 8.4 * (w1 ** 3) + 2 * (w1 ** 5) + w2
        tmp_2 = w1 - 8 * w2 + 16 * (w2 ** 3)
        return [tmp_1,tmp_2]

class DixonPrice(models.Model):
    def __init__(self, name="Dixon-Price",err=0.0):
        super(DixonPrice, self).__init__(name=name)
        self.err = err

    def f_opt(self,w):
        w = np.array(w)
        d = w.shape[0]
        tmp = np.zeros(w.shape)
        for i in range(1,d):
            tmp += (i+1) * (2 * w[i] ** 2  - w[i-1]) ** 2
        tmp += (w[0]-1) ** 2

        return tmp[0]

    def g_opt(self,w):
        w = np.array(w)
        d = w.shape[0]
        tmp = np.zeros(w.shape)
        for i in range(d):
            if i == 0:
                tmp[i] = 2 * (w[i] - 1) + 2 * 2 * (2 * w[i+1] ** 2 - w[i])
            elif i == d-1:
                tmp[i] = 8 * w[i] * d * (i + 1) * (2 * w[i] ** 2 - w[i-1])
            else:
                tmp[i] =  8 * w[i] * d * (i + 1) * (2 * w[i] ** 2 - w[i-1]) - (i + 2) * 2 * (2 * w[i+1] ** 2 - w[i])

        return tmp


class RosenBrock(models.Model):
    def __init__(self, name="RosenBrock",err=0.0):
        super(RosenBrock, self).__init__(name=name)
        self.err = err

    def f_opt(self,w):
        w = np.array(w)
        d = w.shape[0]
        tmp = np.zeros(w.shape)
        for i in range(0, d - 1):
            tmp_1 = 100 * (w[i+1] - w[i] ** 2) ** 2
            tmp_2 = (w[i] -1) ** 2
            tmp += tmp_1 + tmp_2
        return tmp[0]

    def g_opt(self,w):
        w = np.array(w)
        d = w.shape[0]
        tmp = np.zeros(w.shape)
        for i in range(d):
            if i == 0:
                tmp[i] = 100 * (-4) * w[i] * (w[i+1] - w[i] ** 2) + 2 * (w[i] - 1)
            elif i == d-1:
                tmp[i] = 100 * 2 * (w[i] - w[i-1] ** 2)
            else:
                tmp[i] = 100 * (-4) * w[i] * (w[i+1] - w[i] ** 2) + 2 * (w[i] - 1) + 100 * 2 * (w[i] - w[i-1] ** 2)

        return tmp
