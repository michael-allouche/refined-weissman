from extreme.estimators import evt_estimators


# gamma = {0.125, 0.25, 0.5, 1.}
# rho = {-0.125, -0.25, -0.5, -1., -2.}

if __name__ == "__main__":
    LOI = "student"
    RHO = [-2.]

    df_evt = evt_estimators(n_replications=500, params={"evi": 0.125},
                            distribution=LOI, n_data=500, metric="mean", return_full=False)
    df_evt = evt_estimators(n_replications=500, params={"evi": 0.25},
                            distribution=LOI, n_data=500, metric="mean", return_full=False)
    df_evt = evt_estimators(n_replications=500, params={"evi": 0.5},
                            distribution=LOI, n_data=500, metric="mean", return_full=False)
    df_evt = evt_estimators(n_replications=500, params={"evi": 1.},
                            distribution=LOI, n_data=500, metric="mean", return_full=False)


    # df_evt = evt_estimators(n_replications=500, params={"evi": 0.125, "rho": RHO},
    #                         distribution=LOI, n_data=500, metric="mean", return_full=False)
    # df_evt = evt_estimators(n_replications=500, params={"evi": 0.25, "rho": RHO},
    #                         distribution=LOI, n_data=500, metric="mean", return_full=False)
    # df_evt = evt_estimators(n_replications=500, params={"evi": 0.5, "rho": RHO},
    #                         distribution=LOI, n_data=500, metric="mean", return_full=False)
    # df_evt = evt_estimators(n_replications=500, params={"evi": 1., "rho": RHO},
    #                         distribution=LOI, n_data=500, metric="mean", return_full=False)

