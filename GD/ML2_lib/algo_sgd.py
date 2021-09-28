import numpy as np


class Iterative:
    def __init__(self, w_init, t_max=None):

        self.w = np.array(w_init)
        self.t_max = t_max
        self.t = 0
        self.wstore = []

    def __iter__(self):
        return self

    def __next__(self):
        self.t += 1
        if self.t > self.t_max+1:
            raise StopIteration
        else:
            self.wstore.append(self.w)


class SGD(Iterative):

    def __init__(self, w_init, a, fixed_lr=True,t_max=None, data=None):
        super().__init__(w_init, t_max)

        self.a = a
        self.data = data
        self.fixed_lr = fixed_lr

    # 入力は.gという勾配を算出することのできるモデル
    def update(self, model):
        if model.type == "loss_with_data":
            x, y = self.data[0][self.t], self.data[1][self.t]
            grad = model.g(x=x, w=self.w, y=y)
            if self.fixed_lr:
                self.w = self.w - self.a * grad
            else:
                try:
                    self.w = self.w - self.a * grad / self.t
                except:
                    self.w = self.w - self.a * grad
        else:
            grad = model.g_opt(w=self.w)
            if 100000000 < np.sum(grad ** 2):
                grad *= 0.001
            if self.fixed_lr:
                self.w = self.w - self.a * grad
            else:
                try:
                    self.w = self.w - self.a * grad / self.t
                except:
                    self.w = self.w - self.a * grad
