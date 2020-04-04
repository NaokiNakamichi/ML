import numpy as np

class Iterative:
    def __init__(self,w_init,t_max=None):

        self.w = np.array(w_init)
        self.t_max = t_max
        self.t = 0
        self.wstore = []
        self.w_mean = []

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


class LineSearch(Iterative):

    def __init__(self,w_init,a,t_max=None):

        super().__init__(w_init,t_max)

        self.a = a


    def update(self,model,data=None):
        newdir = self.newdir(model=model, data=data)
        self.w = self.w - self.a * newdir
