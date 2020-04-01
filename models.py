import numpy as np

class Model():
    def __init__(self, name="",err=0):

        self.name = name
        self.err = err

    def add_noise(self,x):
        x = np.array(x)
        if x.shape:
            d = x.shape[0]
            x = x + self.err * np.random.randn(d)
        else:
            x = x + self.err * np.random.randn(1)

        return x
