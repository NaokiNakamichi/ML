import numpy as np
import random

class Model():
    def __init__(self, name="",err=0):

        self.name = name
        self.err = err

    def add_noise(self,w):
        return self.err * (np.random.random_sample(w.shape) * random.expovariate(10) )
