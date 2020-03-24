import numpy as np

class GD:
    def __init__(self, w,t_max):

        self.w = w
        self.t_max = t_max
        self.t = 0
        self.wstore = []
        self.loss_store = []

    def __iter__(self):
        return self

    def __next__(self):
        self.t += 1
        if self.t > self.t_max:
            raise StopIteration
        else:
            self.wstore.append(self.w)

    def update(self,model,data=None,a=0.1):
        self.w = self.w - a * model.g_opt(w = self.w)

    def loss(self,X,y):
        tmp = X.dot(self.w) - y
        loss = 0.5 * np.mean(tmp ** 2)
        self.loss_store.append(loss)
