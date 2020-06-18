import numpy as np
import random

class Model():
    def __init__(self,noise ,name="", err=0):

        self.name = name
        self.err = err  #使わなくなりそうな気がする
        self.noise = noise

    def add_noise(self,w):
        
        if self.noise == None:
            return np.zeros(w.shape)
        return self.noise(w)
