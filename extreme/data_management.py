from .distributions import Burr, Frechet, InverseGamma, Fisher, GPD, Student, NHW
import numpy as np

dict_distributions = {"burr": Burr, "invgamma": InverseGamma, "frechet": Frechet, "fisher": Fisher, "gpd": GPD,
                      "student": Student, "nhw": NHW}


def load_distribution(name_distribution):
    return dict_distributions[name_distribution]


class DataSampler():
    def __init__(self, distribution, params, percentile=0, **kwargs):
        self.distribution = distribution
        self.params = params
        self.ht_dist = load_distribution(distribution)(**params)  # heavy-tailed distribution
        self.percentile = percentile

        return

    def simulate_quantiles(self, n_data, low_bound=0., high_bound=1., random=True, seed=32, **kwargs):
        """
        simulate from quantile function  q
        quantiles(random=False) or order statistics (random=True) from heavy-tailed distribution
        Parameters
        ----------
        n_data : int
        low_bound : float
        high_bound : float
        random : bool
            if true, drawn u values from a uniform distribution, else from a linear grid

        Returns
        -------

        """
        if random:
            np.random.seed(seed)
            u_values = np.random.uniform(low_bound, high_bound, size=(int(n_data), 1))  # sample from U( [0, 1) )
            quantiles = np.float32(self.ht_dist.ppf(u_values))
            return np.sort(quantiles, axis=0)  # sort the order statistics
        else:
            u_values = np.linspace(low_bound, high_bound, int(n_data)).reshape(-1,1)
            return np.float32(self.ht_dist.ppf(u_values))







