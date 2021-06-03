import numpy as np


class LinearQuadraticLoss():
    def __init__(self):
        pass

    def f(self, y, x, w):
        return 0.5 * (y - np.dot(w, x)) ** 2

    def g(self, y, x, w):
        # print("y:".format(y))
        #
        # print("w * x".format(w * x))
        return - x * (y - np.dot(w,x))

    def predict(self, x, w):
        return np.dot(w, x)
