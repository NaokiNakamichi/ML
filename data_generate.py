import numpy as np

class Data:
    def __init__(self,d=1,N=100,err=0):

        self.d = d
        self.N = N
        self.err = err

    def generate_data(self):
        w = np.ones(self.d) / self.d
        X = np.random.randn(self.N, self.d)
        y = X.dot(w) + self.err * np.random.randn(self.N)
        init_w = w + np.random.randn(self.d)
        return X,y,init_w,w
