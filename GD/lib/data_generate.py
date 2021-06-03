import numpy as np

class Data:
    def __init__(self,d=1,N=100,err=0):

        self.d = d
        self.N = N
        self.err = err

    def generate_linear(self):
        w_star = np.ones(self.d) / self.d
        X = np.random.randn(self.N, self.d)
        y = X.dot(w_star) + self.err * np.random.randn(self.N)
        return X,y,w_star
