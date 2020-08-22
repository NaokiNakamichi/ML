import numpy as np

class TailAvarageing():
    def __init__(self, w_store,s,batchsize=1):

        self.batchsize = batchsize
        self.w_store = w_store
        self.s = s

    def avraging(self):

        w = np.mean(self.w_store[-self.s:],axis=0)

        return w