import numpy as np


class MinibatchDoublingPartialAveragingSGD:
    def __init__(self,w,n,t,batchsize=1):
        self.batchsize = batchsize
        self.w = w
        self.n = n
        self.t = t
        self.w_store = []

    def calculate(self, algo, model, noise_data):
        s = self.n / (self.batchsize * self.t)




        algo = algo.MinibatchSGD(w_init=self.w, t_max=t, a=0.1, batchsize=self.batchsize)
        for i in algo:
            f = model.Bohachevsky(noise_value=noise_data[algo.t:algo.t + 100])
            algo.update(model=f)

        w = np.mean(self.w_store[-self.s:],axis=0)


        return w