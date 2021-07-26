import numpy as np


class LinearQuadraticLoss():
    def __init__(self):
        self.type = "loss"

    def f(self, y, x, w):

        if type(w) == np.float_:
            return np.mean(0.5 * ((y - np.dot(x, w)) ** 2),axis=0)
        elif type(w) == np.ndarray:
            return np.mean(0.5 * ((y - np.dot(x, w)) ** 2),axis=0)
        else:
            raise ValueError('w のデータ型')

    def g(self, y, x, w):
        # print("y:".format(y))
        #
        # print("w * x".format(w * x))
        return - x * (y - np.dot(w, x))

    def predict(self, x, w):
        return np.dot(w, x)

    def excess_risk_normal(self, X_mean, X_var, w_star, w):
        E_X = np.diag(np.ones(w.shape[0]) * (X_var ** 2) + (X_mean ** 2))
        excess_risk = np.dot(np.dot(E_X, w), w.T) + np.dot(np.dot(E_X, w_star), w_star.T) - 2 * np.dot(
            np.dot(E_X, w_star), w.T)

        return excess_risk[0][0]
