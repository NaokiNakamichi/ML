import numpy as np

import algos

class GD(algos.LineSearch):
    def __init__(self,w_init,t_max,a=0.1):

        super().__init__(w_init=w_init,t_max=t_max,a=a)


    def newdir(self,model,data=None):

        grad = model.g_opt(w = self.w)
        return grad


class SGD(algos.LineSearch):
    def __init__(self,w_init,t_max,a=0.1):

        super().__init__(w_init=w_init,t_max=t_max,a=a)


    def newdir(self,model,data=None):

        grad = model.g_opt(w = self.w)
        return grad
