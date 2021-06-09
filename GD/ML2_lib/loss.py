import numpy as np


class LinearQuadraticLoss():
    def __init__(self):
        pass

    def f(self, y, x, w):

        if type(w) == np.float_:
            return 0.5 * (y - np.dot(w, x)) ** 2
        elif type(w) == np.ndarray:
            if w.shape[0] == 1:
                w = w[0]
                return 0.5 * (y - np.dot(w, x)) ** 2
            else:
                return 0.5 * (y - np.dot(w, x)) ** 2
        else:
            raise ValueError('w のデータ型')

    def g(self, y, x, w):
        # print("y:".format(y))
        #
        # print("w * x".format(w * x))
        return - x * (y - np.dot(w,x))

    def predict(self, x, w):
        return np.dot(w, x)
