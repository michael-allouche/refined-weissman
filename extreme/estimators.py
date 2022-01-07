import numpy as np
import pandas as pd
import scipy.stats
from rpy2 import robjects as ro
import rpy2.robjects.numpy2ri
# from rpy2.robjects.packages import importr
import rpy2.robjects.packages as rpackages

from extreme.data_management import DataSampler
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt


list_estimators = ["W", "RW", "CW", "CH", "CHp", "CHps", "PRBp", "PRBps"]
# list_estimators = ["W", "RW", "CW", "CH", "CHps", "PRBps"]
# list_estimators = ["W", "RW", "CW", "CH"]

# def hill(X_orders, k_anchor):
#     """
#     compute hill estimator
#     Parameters
#     ----------
#     X_orders : order statistics
#         arr
#     k_anchor : anchor point
#         int >=1 and < nb of data
#
#     Returns
#     -------
#
#     """
#     n_data = X_orders.shape[0]
#     X_i_n = X_orders[-k_anchor:]
#     X_k_n = X_orders[n_data-(k_anchor+1)]
#     return np.mean(np.log(X_i_n)) - np.log(X_k_n)

# ==================================================
#                  Tail index estimator
# ==================================================

# def hill(X, k_anchor, axis=0):
#     """
#
#     Parameters
#     ----------
#     X : ndarray
#         arr
#     k : threshold
#         int =>1
#
#     Returns
#     -------
#
#     """
#     X_in = X[-k_anchor:]
#     X_kn = X[-(k_anchor+1)] * np.ones_like(X_in)
#
#     return np.mean(np.log(X_in) - np.log(X_kn))


class TailIndexEstimator():
    def __init__(self, X_order):
        """initialization of the hyperparameters with specific methods"""
        # R settings
        rpy2.robjects.numpy2ri.activate()
        r = ro.r
        r['source']('extreme/revt.R')
        self.get_rho_beta = r.get_rho_beta


        self.X_order = X_order
        self.n_data = X_order.shape[0]

        self.rho, self.beta = self.get_rho_beta(X_order)  # rho and beta estimated
        self.varphi = 1 - self.rho/2 - np.sqrt(np.square(1 - self.rho/2) - 0.5)
        self.k0 = self.get_k0()
        self.p_star = self.varphi / self.corrected_hill(self.k0)
        return

    def hill(self, k_anchor):
        """H"""
        X_in = self.X_order[-k_anchor:]
        X_kn = self.X_order[-(k_anchor + 1)] * np.ones_like(X_in)  # take also the last point X_n,n
        return np.mean(np.log(X_in) - np.log(X_kn))

    def corrected_hill(self, k_anchor):
        """CH"""
        gamma_hill = self.hill(k_anchor)
        return gamma_hill * (1 - (self.beta / (1 - self.rho) * np.power(self.n_data / k_anchor, self.rho)))

    def hill_p(self, k_anchor, p):
        """H_p"""
        if p == 0.:
            gamma = self.hill(k_anchor)
        else:
            X_in = self.X_order[-k_anchor:]
            X_kn = self.X_order[-(k_anchor + 1)] * np.ones_like(X_in)
            gamma = (1 - np.power(np.mean(np.power(X_in / X_kn, p)), -1)) / p
        return gamma


    def corrected_hill_p(self, k_anchor, p=None):
        """CH_p"""
        if p is None:
            p = self.p_CH
        gamma = self.hill_p(k_anchor, p)
        return gamma * (1 - ((self.beta * (1 - p * gamma)) / (1 - self.rho - p * gamma) * np.power(self.n_data / k_anchor, self.rho)))

    def corrected_hill_ps(self, k_anchor):
        """CH_ps"""
        gamma = self.hill_p(k_anchor, self.p_star)
        return gamma * (1 - ((self.beta * (1 - self.p_star * gamma)) / (1 - self.rho - self.p_star * gamma) * np.power(self.n_data / k_anchor, self.rho)))

    def partially_reduced_bias_p(self, k_anchor, p=None):
        "PRB_p"
        if p is None:
            p = self.p_PRB
        gamma = self.hill_p(k_anchor, p)
        return gamma * (1 - ((self.beta * (1 - self.varphi)) / (1 - self.rho - self.varphi) * np.power(self.n_data / k_anchor, self.rho)))

    def partially_reduced_bias_ps(self, k_anchor):
        "PRB_ps"
        gamma = self.hill_p(k_anchor, self.p_star)
        return gamma * (1 - ((self.beta * (1 - self.varphi)) / (1 - self.rho - self.varphi) * np.power(self.n_data / k_anchor, self.rho)))

    def get_k0(self):
        term1 = self.n_data - 1
        term2 = np.power(np.square(1 - self.rho) * np.power(self.n_data, -2*self.rho) / (-2*self.rho*np.square(self.beta)), 1/(1-2*self.rho))
        return int(np.minimum(term1, np.floor(term2) + 1))



# ==================================================
#                  Extreme quantile estimators
# ==================================================


# def weissman(X_order, alpha, k_anchor):
#     """
#     Parameters
#     ----------
#     X_orders : order statistics
#     alpha : extreme order
#     k_anchor : anchor point
#
#     Returns
#     -------
#
#     Maths
#     ----
#     X_{n-k, n}(k/np)^gamma_hill(k) with 0<p<1 and k\in{1,...,n-1}
#
#     """
#     gamma_hill = hill(X_order, k_anchor)
#     n_data = X_order.shape[0]
#     X_anchor = X_order[-k_anchor]
#     return X_anchor * np.power(k_anchor/(alpha * n_data), gamma_hill)


class ExtremeQuantileEstimator(TailIndexEstimator):
    def __init__(self, X, alpha):
        """
        Parameters
        ----------
        X : ndarray
            order statistics X_{1,n}, ..., X_{n,n}
        alpha : float
            extreme order
        """
        super(ExtremeQuantileEstimator, self).__init__(X)
        self.alpha = alpha
        self.dict_q_estimators = {"W": self.weissman, "RW": self.r_weissman, "CW": self.c_weissman,
                                  "CH": self.ch_weissman, "CHps": self.chps_weissman,
                                   "PRBps": self.prbps_weissman}
        self.dict_qp_estimators = {"CHp": self.chp_weissman, "PRBp": self.prbp_weissman}
        self.dict_quantile_estimators = {**self.dict_q_estimators, **self.dict_qp_estimators}

        self.p_CH = self.get_p(method="CHp")
        self.p_PRB = self.get_p(method="PRBp")
        return

    def weissman(self, k_anchor):
        """Weissman (W)"""
        X_anchor = self.X_order[-k_anchor]  # X_{n-k+1, n} for k=2,..., n-1
        extrapolation_ratio = k_anchor / (self.alpha * self.n_data)
        return X_anchor * np.power(extrapolation_ratio, self.hill(k_anchor))

    def r_weissman(self, k_anchor):
        """Revisited Weissman (RW)"""
        # mu = 1 / (1-self.rho)
        X_anchor = self.X_order[-k_anchor]  # X_{n-k+1, n} for k=2,..., n-1
        extrapolation_ratio = k_anchor / (self.alpha * self.n_data)
        k_prime = k_anchor * np.power((-self.rho * np.log(extrapolation_ratio)) / ((1-self.rho) * (1 - np.power(extrapolation_ratio, self.rho))), 1/self.rho)
        return X_anchor * np.power(extrapolation_ratio, self.hill(int(np.ceil(k_prime))))

    def c_weissman(self, k_anchor):
        """Corrected Weissman (CW)"""
        X_anchor = self.X_order[-k_anchor]  # X_{n-k+1, n} for k=2,..., n-1
        extrapolation_ratio = k_anchor / (self.alpha * self.n_data)
        return X_anchor * np.power(extrapolation_ratio *
                                        np.exp(self.beta * np.power(self.n_data/k_anchor, self.rho)
                                               * (np.power(extrapolation_ratio, self.rho) - 1) / self.rho), self.corrected_hill(k_anchor))

    def ch_weissman(self, k_anchor):
        """Corrected-Hill Weissman (CH)"""
        X_anchor = self.X_order[-k_anchor]  # X_{n-k+1, n} for k=2,..., n-1
        extrapolation_ratio = k_anchor / (self.alpha * self.n_data)
        return X_anchor * np.power(extrapolation_ratio, self.corrected_hill(k_anchor))

    def chp_weissman(self, k_anchor, p=None):
        """Corrected-Hill with Mean-of-order-p Weissman (CHp)"""
        X_anchor = self.X_order[-k_anchor]  # X_{n-k+1, n} for k=2,..., n-1
        extrapolation_ratio = k_anchor / (self.alpha * self.n_data)
        return X_anchor * np.power(extrapolation_ratio, self.corrected_hill_p(k_anchor, p))

    def chps_weissman(self, k_anchor):
        """Corrected-Hill with Mean-of-order-p star (optimal) Weissman (CHps)"""
        X_anchor = self.X_order[-k_anchor]  # X_{n-k+1, n} for k=2,..., n-1
        extrapolation_ratio = k_anchor / (self.alpha * self.n_data)
        return X_anchor * np.power(extrapolation_ratio, self.corrected_hill_ps(k_anchor))

    def prbp_weissman(self, k_anchor, p=None):
        """Partially Reduced-Bias mean-of-order-p Weissman (PRBp)"""
        X_anchor = self.X_order[-k_anchor]  # X_{n-k+1, n} for k=2,..., n-1
        extrapolation_ratio = k_anchor / (self.alpha * self.n_data)
        return X_anchor * np.power(extrapolation_ratio, self.partially_reduced_bias_p(k_anchor, p))

    def prbps_weissman(self, k_anchor):
        """Partially Reduced-Bias mean-of-order-p star (optimal) Weissman (PRBPs)"""
        X_anchor = self.X_order[-k_anchor]  # X_{n-k+1, n} for k=2,..., n-1
        extrapolation_ratio = k_anchor / (self.alpha * self.n_data)
        return X_anchor * np.power(extrapolation_ratio, self.partially_reduced_bias_ps(k_anchor))

    def quantile_estimator(self, method, k_anchor):
        return self.dict_quantile_estimators[method](k_anchor)


    def get_p(self, method):
        """
        get best p and k based on Algo 2 from Gomes, 2018
        Parameters
        ----------
        method :

        Returns
        -------

        """
        # STEP 1
        xi_star = self.corrected_hill(self.k0)
        p_ell = np.arange(16)/(16*xi_star)

        #STEP 2
        list_runsize = []
        for ell in range(16):
            # STEP 2.1:  applied on log(.) as suggested in the paper
            quantiles = np.log([self.dict_qp_estimators[method](k_anchor=k_anchor, p=p_ell[ell])[0] for k_anchor in range(2, self.n_data)])
            # STEP 2.2: find the minimum j>=0 s.t all q_rounded are distinct
            j = 0
            optimal = False
            while not optimal:
                q_rounded = np.around(quantiles, j)
                if np.unique(q_rounded).shape[0] == q_rounded.shape[0]:  # if all uniques
                    optimal = True
                else:
                    j += 1
            # STEP 2.3
            k_min, k_max = self.longest_run(q_rounded, j)
            list_runsize.append(k_max - k_min)
        largest_runsize_idx = np.argmax(list_runsize)
        # STEP 3
        p = largest_runsize_idx / (16*xi_star)
        return p[0]

    @staticmethod
    def longest_run(x, j):
        """
        compute the run size k_min and k_max
        Parameters
        ----------
        x : ndarray
        j: int
            decimal point

        Returns
        -------
        k_min, k_max
        """
        # x = (np.abs(x) * 10**j).astype(int)  # convert to integers and absolute value to remove -
        mat = np.zeros(shape=(len(x), j + 1))
        for idx in range(len(x)):
            for val in range(j):
                # split the integer into array. Add "1"*(j+1) to avoid problem with numbers starting by 0
                mat[idx, val] = int(str(int(float('% .{}f'.format(j)%np.abs(x[idx]))*10**j) + + int("1"*(j+1)))[val])

        diff_mat = np.diff(mat, axis=1)  # diff in between columns
        list_k = np.count_nonzero(diff_mat == 0., axis=1)  # count number of zeros in columns
        return np.min(list_k), np.max(list_k)



def tree_k(x, a=None, c=None, return_var=False):
    """
    choice of the best k based on the dyadic decomposition.
    returns the Python index (starts at 0). Add 2 to get the order level.
    """
    if a is None:
        a = 13
    if c is None:
        c = int(3*x.shape[0] / 4)
    b = int((c+a)/2)

    list_var = []
    finish = False
    while not finish:
        if (b-a) < 2:
            finish = True
        else:
            v1 = np.var(x[a:b+1])
            v2 = np.var(x[b:c+1])
            if v1 < v2:  # left wins
                list_var.append(v1)
                c = b
            else:  # right wins
                list_var.append(v2)
                a = b
            b = int((c + a) / 2)
    if return_var:
        return b, np.mean(list_var)
    return b


def random_forest_k(x, n_forests, seed=42):
    np.random.seed(seed)
    a0 = 13
    c0 = int(3 * x.shape[0] / 4)
    list_k = []

    for i in range(n_forests):
        a = np.random.randint(a0, c0)
        c = np.random.randint(a+1, c0+1)
        list_k.append(tree_k(x, a, c))

    return int(np.median(np.array(list_k)))





def evt_estimators(n_replications, n_data, distribution, params, metric, return_full=False):
    dict_evt = {estimator: {_metric: {"series": [], "rmse_bestK": None, "q_bestK": [],
                            "bestK": []}for _metric in ["mean", "median"]} for estimator in list_estimators}

    pathdir = Path("ckpt", distribution, "extrapolation", str(params))
    pathdir.mkdir(parents=True, exist_ok=True)

    try:
        dict_evt = np.load(Path(pathdir, "evt_estimators_rep{}.npy".format(n_replications)), allow_pickle=True)[()]
    except FileNotFoundError:
        anchor_points = np.arange(2, n_data)  # 2, ..., n-1
        EXTREME_ALPHA = 1 / (2 * n_data)  # extreme alpha
        data_sampler = DataSampler(distribution=distribution, params=params)
        real_quantile = data_sampler.ht_dist.tail_ppf(1 / EXTREME_ALPHA)  # real extreme quantile

        for replication in range(1, n_replications + 1):  # for each replication
            X_order = data_sampler.simulate_quantiles(n_data, seed=replication, random=True).reshape(-1, 1)  # order statistics X_1,n, ..., X_n,n
            dict_q = {estimator: [] for estimator in list_estimators}  # dict of quantiles
            evt_estimators = ExtremeQuantileEstimator(X=X_order, alpha=EXTREME_ALPHA)

            for estimator in list_estimators:
                for anchor_point in anchor_points:  # compute all quantile estimators
                    dict_q[estimator].append(evt_estimators.quantile_estimator(k_anchor=anchor_point, method=estimator)[0])

                bestK = random_forest_k(np.array(dict_q[estimator]), 10000)

                # MEAN
                dict_evt[estimator]["mean"]["series"].append(dict_q[estimator])
                dict_evt[estimator]["mean"]["q_bestK"].append(dict_q[estimator][int(bestK)])
                dict_evt[estimator]["mean"]["bestK"].append(bestK + 2)  # k \geq 2

                # MEDIAN
                dict_evt[estimator]["median"]["series"].append(dict_q[estimator])
                dict_evt[estimator]["median"]["q_bestK"].append(dict_q[estimator][int(bestK)])
                dict_evt[estimator]["median"]["bestK"].append(bestK + 2)  # k \geq 2


        for estimator in list_estimators:
            # MEAN
            dict_evt[estimator]["mean"]["var"] = np.array(dict_evt[estimator]["mean"]["series"]).var(axis=0)
            dict_evt[estimator]["mean"]["rmse"] = ((np.array(dict_evt[estimator]["mean"]["series"]) / real_quantile - 1) ** 2).mean(axis=0)
            dict_evt[estimator]["mean"]["series"] = np.array(dict_evt[estimator]["mean"]["series"]).mean(axis=0)
            dict_evt[estimator]["mean"]["rmse_bestK"] = ((np.array(dict_evt[estimator]["mean"]["q_bestK"]) / real_quantile - 1) ** 2).mean()

            # MEDIAN
            dict_evt[estimator]["median"]["var"] = np.array(dict_evt[estimator]["median"]["series"]).var(axis=0)
            dict_evt[estimator]["median"]["rmse"] = np.median((np.array(dict_evt[estimator]["median"]["series"]) / real_quantile - 1) ** 2, axis=0)
            dict_evt[estimator]["median"]["series"] = np.median(dict_evt[estimator]["median"]["series"], axis=0)
            dict_evt[estimator]["median"]["rmse_bestK"] = np.median((np.array(dict_evt[estimator]["median"]["q_bestK"]) / real_quantile - 1) ** 2)


        np.save(Path(pathdir, "evt_estimators_rep{}.npy".format(n_replications)), dict_evt)

    if return_full:
        return dict_evt
    df = pd.DataFrame(columns=list_estimators, index=["RMSE"])
    for estimator in list_estimators:
        df.loc["RMSE", estimator] = dict_evt[estimator][metric]["rmse_bestK"]
    return df


