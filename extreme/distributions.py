import numpy as np
from scipy import stats

class FrechetMDA2OC():
    def __init__(self):
        self.evi = None  # extreme value index
        self.rho = None  # J order parameters
        return

    def cdf(self, x):
        raise ("No distribution called")

    def sf(self, x):
        """survival function """
        return 1 - self.cdf(x)

    def ppf(self, u):
        """quantile function"""
        raise ("No distribution called")

    def isf(self, u):
        """inverse survival function"""
        return self.ppf(1 - u)

    def tail_ppf(self, x):
        """tail quantile function U(x)=q(1-1/x)"""
        return self.isf(1/x)

    def norm_ppf(self, u):
        "quantile normalized X>=1"
        return self.isf((1 - u) * self.sf(1))


class Burr(FrechetMDA2OC):
    def __init__(self, evi, rho):
        super(Burr, self).__init__()
        self.evi = evi
        self.rho = np.array(rho)
        return

    def cdf(self, x):
        return 1 - (1 + x ** (- self.rho / self.evi)) ** (1 / self.rho)

    def ppf(self, u):
        return (((1 - u) ** self.rho) - 1) ** (- self.evi / self.rho)


class InverseGamma(FrechetMDA2OC):
    def __init__(self, evi):
        super(InverseGamma, self).__init__()
        self.evi = evi
        self.rho = np.array(-self.evi)
        self.law = stats.invgamma(1/self.evi)
        return
    def cdf(self, x):
        return self.law.cdf(x)

    def ppf(self, u):
        return self.law.ppf(u)

class Frechet(FrechetMDA2OC):
    def __init__(self, evi):
        super(Frechet, self).__init__()
        self.evi = evi
        self.rho = np.array([-1.])
        self.law = stats.invweibull(1 / self.evi)
        return

    def cdf(self, x):
        return self.law.cdf(x)

    def ppf(self, u):
        return self.law.ppf(u)

class Fisher(FrechetMDA2OC):
    def __init__(self, evi):
        super(Fisher, self).__init__()
        self.evi = evi
        self.rho = np.array([-2./self.evi])
        self.law = stats.f(3, 2/self.evi)
        return
    def cdf(self, x):
        return self.law.cdf(x)

    def ppf(self, u):
        return self.law.ppf(u)


class GPD(FrechetMDA2OC):
    def __init__(self, evi):
        super(GPD, self).__init__()
        self.evi = evi
        self.rho = np.array([-self.evi])
        self.law = stats.genpareto(self.evi)
        return
    def cdf(self, x):
        return self.law.cdf(x)

    def ppf(self, u):
        return self.law.ppf(u)


class Student(FrechetMDA2OC):
    def __init__(self, evi):
        super(Student, self).__init__()
        self.evi = evi
        self.rho = np.array([-2*self.evi])
        self.law = stats.t(1/self.evi)
        return

    def cdf(self, x):
        return 2 * self.law.cdf(x) - 1

    def ppf(self, u):
        return self.law.ppf((u+1)/2)

class NHW(FrechetMDA2OC):
    def __init__(self, evi, rho):
        super(NHW, self).__init__()
        self.evi = evi
        self.rho = np.array(rho)
        
    def ppf(self, u):
        t = 1 / (1-u)
        A = self.rho * (t ** self.rho) * np.log(t) / 2
        return np.power(t, self.evi) * np.exp(A / self.rho)



if __name__ == '__main__':
    import matplotlib.pyplot as plt
    evi = 0.5
    rho = [-1]  # \bar rho_j order parameter
    n_data = 100
    u = np.linspace(0, 1-1/n_data, n_data).reshape(-1, 1)

    ht = Burr(evi, rho)
    quantiles = ht.ppf(u)
    plt.plot(u, quantiles)
    plt.show()

