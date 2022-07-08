from extreme.estimators import evt_estimators
from itertools import product
import argparse

dict_runner = {"burr": {"evi": [0.125, 0.25, 0.5, 1.], "rho": [-0.125, -0.25, -0.5, -1.]},
               "nhw": {"evi": [0.125, 0.25, 0.5, 1.], "rho": [-0.125, -0.25, -0.5, -1.]},
               "frechet": {"evi": [0.125, 0.25, 0.5, 1.]},
               "fisher": {"evi": [0.125, 0.25, 0.5, 1.]},
               "gpd": {"evi": [0.125, 0.25, 0.5, 1.]},
               "invgamma": {"evi": [0.125, 0.25, 0.5, 1.]},
               "student": {"evi": [0.125, 0.25, 0.5, 1.]}
               }


parser = argparse.ArgumentParser(description='Runner for simulations')
parser.add_argument('--distribution', '-d', type=str,
                    help="name of the dstribution: {burr, nhw, frechet, fisher, gpd, invgamma, student} ",
                    default="burr")
parser.add_argument('--replications', '-r', type=int,
                    help="Number of replications",
                    default=1000)
parser.add_argument('--ndata', '-n', type=int,
                    help="Number of observations ",
                    default=500)
parser.add_argument('--level', '-l', type=str,
                    help="level quantile: {'n', '2n'} ",
                    default="2n")



if __name__ == "__main__":
    args = parser.parse_args()
    distribution = args.distribution
    n_replications = args.replications
    n_data = args.ndata
    level_quantile = args.level

    keys, values = zip(*dict_runner[distribution].items())
    permutations_dicts = [dict(zip(keys, v)) for v in product(*values)]
    for parameters in permutations_dicts:
        print("{}: ".format(distribution),parameters)
        df_evt = evt_estimators(n_replications=n_replications, params=parameters,
                                distribution=distribution, n_data=n_data, n_quantile=level_quantile)



