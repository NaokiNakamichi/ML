import numpy as np


class Noise:
    def __init__(self, dim=1, mean=0, sigma=1, n=1):
        self.dim = dim
        self.n = n
        self.mean = mean
        self.sigma = sigma

    def normal(self):
        rng = np.random.default_rng()
        if self.n == 1:
            return rng.normal(loc=self.mean, scale=self.sigma, size=self.dim)

        return rng.normal(loc=self.mean, scale=self.sigma, size=(self.dim, self.n))

    def lognormal(self):
        rng = np.random.default_rng()
        if self.n == 1:
            sample = rng.lognormal(mean=self.mean, sigma=self.sigma, size=self.dim)
            pop_mean = np.exp(self.mean + (self.sigma ** 2) / 2)
            return sample - pop_mean
        else:
            sample = rng.lognormal(mean=self.mean, sigma=self.sigma, size=(self.dim, self.n))
            pop_mean = np.exp(self.mean + (self.sigma ** 2) / 2)
            return sample - pop_mean


