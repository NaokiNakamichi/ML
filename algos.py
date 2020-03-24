import numpy as np

class GD:
    def __init__(self, w,t_max):

        self.w = np.array(w)
        self.t_max = t_max
        self.t = 0
        self.wstore = []

    def __iter__(self):
        return self

    def __next__(self):
        self.t += 1
        if self.t > self.t_max:
            raise StopIteration
        else:
            self.wstore.append(self.w)

    def update(self,a,grad):
        self.w = self.w - a * grad
