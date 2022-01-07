from extreme.estimators import evt_estimators
from itertools import product

dict_runner = {"burr": {"evi": [0.125, 0.25, 0.5, 1.], "rho": [-0.125, -0.25, -0.5, -1.]},
               "nhw": {"evi": [0.125, 0.25, 0.5, 1.], "rho": [-0.125, -0.25, -0.5, -1.]},
               "frechet": {"evi": [0.125, 0.25, 0.5, 1.]},
               "fisher": {"evi": [0.125, 0.25, 0.5, 1.]},
               "gpd": {"evi": [0.125, 0.25, 0.5, 1.]},
               "invgamma": {"evi": [0.125, 0.25, 0.5, 1.]},
               "student": {"evi": [0.125, 0.25, 0.5, 1.]}
               }

if __name__ == "__main__":
    DISTRIBUTION = "student"
    keys, values = zip(*dict_runner[DISTRIBUTION].items())
    permutations_dicts = [dict(zip(keys, v)) for v in product(*values)]
    for parameters in permutations_dicts:
        df_evt = evt_estimators(n_replications=1000, params=parameters,
                                distribution=DISTRIBUTION, n_data=500, metric="mean")


