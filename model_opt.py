import numpy as np

import models

class Bohachevsky(models.Model):

    def __init__(self, name="Bohachevsky",err=0.0):
        super(Bohachevsky, self).__init__(name=name)
        self.err = err

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
        w1,w2 = w[0],w[1]
        g_w1 = 2 * w1 + 0.9 * np.pi * np.sin(3 * np.pi * w1)
        g_w2 = 4 * w2 + 1.6 * np.pi * np.sin(4 * np.pi * w2)
        dim = w.ndim
        shape = w.shape
        if dim == 0:
            g_w1 = g_w1 + self.err * np.random.randn(1)
            g_w2 = g_w2 + self.err * np.random.randn(1)
            g = np.array([g_w1,g_w2])
            return g

        elif dim == 1:
            g_w1 = g_w1 + self.err * np.random.randn(1)
            g_w2 = g_w2 + self.err * np.random.randn(1)
            g = np.array([g_w1,g_w2]).flatten()
            return g

        else:
            g_w1 = g_w1 + self.err * np.random.randn(g_w1.shape[0],g_w1.shape[1])
            g_w2 = g_w2 + self.err * np.random.randn(g_w1.shape[0],g_w1.shape[1])
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
        w = 
