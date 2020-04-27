import numpy as np
import random

class Model():
    def __init__(self, name="",err=0):

        self.name = name
        self.err = err

    def add_noise(self,w):
        d = w.shape[0]
        random_err = []
        for i in range(d):
            random_err.append(random.random())

        err = np.array(random_err) * np.random.randn(d)

        return w + self.err * err
