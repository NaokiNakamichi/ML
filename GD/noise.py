import numpy as np


class Noise:
    def __init__(self,dim, n=10):
        self.value_store = []
        self.igr_value = 1
        self.dim = dim
        self.n = n


class Gauss(Noise):
    def __init__(self, dim=1, mean=0, sigma=1, n=10):
        super().__init__(dim=dim, n=n)
        self.mean = mean
        self.sigma = sigma

    def generate(self):
        np.random.seed()
        value = np.random.normal(loc=self.mean, scale=self.sigma, size=(self.n, self.dim))
        self.value_store = value
        return value


class LogNormal(Noise):
    def __init__(self,dim=1,mean=0, sigma=1, n=10):
        super().__init__(dim=dim, n=n)
        self.mean = mean
        self.sigma = sigma

    def generate(self):
        np.random.seed()
        value = np.random.lognormal(mean=self.mean, sigma=self.sigma, size=(self.n, self.dim))
        value = value - np.mean(value, axis=0)
        self.value_store = value
        return value


class StudentT(Noise):
    def __init__(self,dim=1, n=10, df=2):
        super().__init__(dim=dim,n=n)
        self.df = df

    def generate(self):
        np.random.seed()
        value = np.random.standard_t(df=self.df, size=(self.n,self.dim))
        self.value_store = value
        return value


class Pareto(Noise):
    def __init__(self,dim=1, n=10, a=1):
        super().__init__(dim=dim,n=n)
        self.a = a

    def generate(self):
        np.random.seed()
        value = np.random.pareto(a=self.a, size=(self.n,self.dim))
        value = value - np.mean(value, axis=0)
        self.value_store = value
        return value


class GaussianMixture(Noise):
    def __init__(self, dim=1, mean=1, sigma=1, n=10):
        super().__init__(dim=dim, n=n)
        self.mean = mean
        self.sigma = sigma

    def generate(self):
        np.random.seed()
        gauss1 = np.random.normal(loc=self.mean, scale=self.sigma, size=(self.n, self.dim))
        gauss2 = np.random.normal(loc=-self.mean, scale=self.sigma, size=(self.n, self.dim))

        i = np.random.choice(self.dim, self.n, p=[0.5, 0.5]).reshape(1, self.n)
        i_inv = 1 - i

        value = i.T * gauss1 + i_inv.T * gauss2
        value = value - np.mean(value, axis=0)
        self.value_store = value
        return value


class NoNoise(Noise):
    def __init__(self,dim=1):
        super().__init__(dim=dim)

    def generate(self):
        value = np.zeros(self.dim)
        self.value_store.append(value)
        return value

