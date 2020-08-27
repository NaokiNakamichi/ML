import numpy as np
import algo_GD


class MinibatchTailAveragingSGD:

    def __init__(self, a, w, n, s, noise_data, batchsize=1):
        self.batchsize = batchsize
        self.a = a
        self.w = w
        self.n = n
        self.s = s
        self.noise_data = noise_data
        self.w_store = []

    def calculate(self,model_func):
        t_max = int(self.n / self.batchsize)
        algo = algo_GD.MinibatchSGD(w_init=self.w, t_max = t_max, a=self.a)
        self.w_store.append(self.w)

        for i in algo:
            f = model_func(noise_value=self.noise_data[algo.t:algo.t+self.batchsize])
            algo.update(f)
            self.w_store.append(algo.w)

        self.w = np.mean(self.w_store[-self.s:], axis=0)

        return self.w


class MinibatchDoublingPartialAveragingSGD:
    def __init__(self, a, w, n, t, noise_data, init_batchsize=1):
        self.init_batchsize = init_batchsize
        self.a = a
        self.w = w
        self.n = n
        self.t = t
        self.noise_data = noise_data
        self.w_store = []

    def tail(self, model_func):
        self.w_store.append(self.w)

        s = int(np.log2(self.n / (self.init_batchsize * self.t)))

        for l in range(1, s):
            batchsize_l = 2 ** (l - 1) * self.init_batchsize
            algo = MinibatchTailAveragingSGD(a=self.a, w=self.w, batchsize=batchsize_l,\
                                             s=self.t - 1, n=self.t * batchsize_l,noise_data=self.noise_data)
            self.w = algo.calculate(model_func)
            self.w_store.append(self.w)

        algo = MinibatchTailAveragingSGD(a=self.a, w=self.w, batchsize=int(self.n / (2*self.t)), \
                                         s=int(self.t/2), n=int(self.n/2), noise_data=self.noise_data)
        self.w = algo.calculate(model_func)
        self.w_store.append(self.w)

        return self.w


class PolyakRupportAveraging:

    def __init__(self, a, t, w, eta, noise_data):
        self.t = t
        self.a = a
        self.w = w
        self.eta = eta
        self.noise_data = noise_data
        self.w_store = []

    def calculate(self, model_func):
        self.w_store.append(self.w)
        algo = algo_GD.SGD(w_init=self.w, t_max = self.t, a=self.a)
        for i in algo:
            f = model_func(noise_value=self.noise_data[algo.t-1])
            algo.update(f)
            self.w_store.append(algo.w)
            tmp = 0
            total_w = 0
            for k in range(algo.t):
                tmp += (1 - self.eta) ** k
                total_w += ((1 - self.eta) ** k) * algo.wstore[k]
            self.w = total_w / tmp
            self.w_store.append(self.w)

        return self.w



