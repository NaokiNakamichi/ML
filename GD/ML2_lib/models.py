import numpy as np

from torch import nn
import torch.nn.functional as F

class Model():
    def __init__(self,noise=None,name=""):

        self.name = name
        self.noise = noise

    def add_noise(self,w,sigma=1):
        
        if self.noise == None:
            return np.zeros(w.shape)
        return self.noise(w,sigma=sigma)


class LinearClassification(nn.Module):
    def __init__(self,w_num,c_num):
        super(LinearClassification, self).__init__()
        self.fc1 = nn.Linear(w_num, c_num)

    def forward(self, x):
        x = self.fc1(x)
        return F.log_softmax(x, dim=1)

    def parameter_init(self):
        nn.init.constant_(self.fc1.weight, 0)
        nn.init.constant_(self.fc1.bias, 0)