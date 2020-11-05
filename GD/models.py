import numpy as np

class Model():
    def __init__(self,noise=None,name=""):

        self.name = name
        self.noise = noise

    def add_noise(self,w,sigma=1):
        
        if self.noise == None:
            return np.zeros(w.shape)
        return self.noise(w,sigma=sigma)
