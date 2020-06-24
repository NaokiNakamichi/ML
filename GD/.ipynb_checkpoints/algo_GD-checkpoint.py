import numpy as np

import algos

class GD(algos.LineSearch):
    def __init__(self,w_init,t_max,a=0.1):

        super().__init__(w_init=w_init,t_max=t_max,a=a)


    def newdir(self,model,data=None,label=None):
        if data is not None: # modelは二乗誤差のみを想定
            n = label.shape[0]
            tmp = data.dot(self.w) - label
            grad = model.g_opt(w = tmp)
            grad = grad.T.dot(data) / n
            return grad
        else:
            grad = model.g_opt(w = self.w)
            return grad


class SGD(algos.LineSearch):
    def __init__(self,w_init,t_max,a=0.1,batchsize=1):

        super().__init__(w_init=w_init,t_max=t_max,a=a)
        self.batchsize = batchsize


    def newdir(self,model,data=None,label=None):
        if data is not None: # modelは二乗誤差のみを想定
            n = label.shape[0]
            id = np.random.randint(n)
            tmp = data[id].dot(self.w) - label[id]
            grad = model.g_opt(w = tmp)
            grad = grad * data[id]
            return grad
        else:
            grad, noise_value = model.g_opt(w = self.w)
            return grad, noise_value
