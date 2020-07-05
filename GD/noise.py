import numpy as np


class Noise:
    def __init__(self,dim):
        self.value_store = []
        self.igr_value = 1
        self.dim = dim

    def iqr(self):
        q75, q25 = np.percentile(self.value_store, [75, 25])
        iqr_value = q75 - q25
        self.igr_value = iqr_value


class Gauss(Noise):
    def __init__(self,dim=1):
        super().__init__(dim=dim)

    def generate(self,mean=0,sigma=1):
        np.random.seed()
        value = np.random.normal(loc=mean, scale=sigma, size=self.dim)
        self.value_store.append(value)
        return value


class Lognormal(Noise):
    def __init__(self,dim=1):
        super().__init__(dim=dim)

    def generate(self,mean=0,sigma=1):
        np.random.seed()
        value = np.random.lognormal(mean=mean, sigma=sigma, size=self.dim)
        self.value_store.append(value)
        return value


class StudentT(Noise):
    def __init__(self,dim=1):
        super().__init__(dim=dim)

    def generate(self,v=2):
        np.random.seed()
        value = np.random.standard_t(v, size=self.dim)
        self.value_store.append(value)
        return value

