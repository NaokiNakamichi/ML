import numpy as np

class GD:
    def __init__(self,w=1,t_max=100):

        self.w = np.array(w)
        self.t_max = t_max
        self.t = 0
        self.wstore = []
        self.w_mean = []
        self.loss_store = []
        self.f_store = []

    def __iter__(self):
        return self

    def __next__(self):
        self.t += 1
        if self.t > self.t_max:
            raise StopIteration
        else:
            self.wstore.append(self.w)
            tmp = np.sum(self.wstore,axis=0)/self.t
            self.w_mean.append(tmp)


    def update(self,model,data=None,a=0.1):
        if data is not None:
            grad = a * model.g_opt(w = data)
        else:
            grad = a * model.g_opt(w = self.w)
            loss = np.sum((model.w_star - self.w) ** 2)
            self.loss_store.append(loss)
            self.f_store.append(model.f_opt(self.w))

        if model.err != 0:
            grad = model.add_noise(grad)


        self.w = self.w - a * grad





class SGD:
    def __init__(self, w, t_max):

        self.w = w
        self.t_max = t_max
        self.t = 0
        self.w_wtore = []
        self.loss_store = []

    def __iter__(self):
        return self

    #def update(self, model, data)
