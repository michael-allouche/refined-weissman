import numpy as np
import pandas as pd
from extreme.estimators import evt_estimators, real_estimators, list_estimators, ExtremeQuantileEstimator, random_forest_k
from extreme.data_management import DataSampler
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import os


def evt_quantile_plot(n_replications, n_data, distribution, params, n_quantile):
    """extreme quantile plot of just the evt estimatorsat level 1/2n for different replications with variance and MSE"""

    pathdir = Path("ckpt", n_quantile, distribution, "extrapolation", str(params))
    pathdir.mkdir(parents=True, exist_ok=True)

    anchor_points = np.arange(2, n_data)  # 1, ..., n-1
    if n_quantile == "2n":
        EXTREME_ALPHA = 1 / (2 * n_data)  # extreme alpha
    elif n_quantile == "n":
        EXTREME_ALPHA = 1 / (n_data)  # extreme alpha
    else:
        return "The 'n_quantile' doesn't exist. PLese choose between {'n', '2n'}."
        
    data_sampler = DataSampler(distribution=distribution, params=params)
    real_quantile = data_sampler.ht_dist.tail_ppf(1 / EXTREME_ALPHA)  # real extreme quantile

    try:
        dict_evt = np.load(Path(pathdir, "evt_estimators_rep{}.npy".format(n_replications)), allow_pickle=True)[()]
    except FileNotFoundError:
        dict_evt = evt_estimators(n_replications, n_data, distribution, params, n_quantile, return_full=True)

    fig, axes = plt.subplots(3, 1, figsize=(15, 3 * 5), sharex=False, squeeze=False)  # 3 plots: quantile, var, mse

    for estimator in dict_evt.keys():
        axes[0, 0].plot(anchor_points, dict_evt[estimator]["series"],
                        label="{} (rmse={:.2f})".format(estimator, dict_evt[estimator]["rmse_bestK"],
                                                                        ), linestyle="-.")
        axes[1, 0].plot(anchor_points, dict_evt[estimator]["var"],
                        label="{} (rmse={:.2f})".format(estimator, dict_evt[estimator]["rmse_bestK"],
                                                                        ), linestyle="-.")
        axes[2, 0].plot(anchor_points, dict_evt[estimator]["rmse"],
                        label="{} (rmse={:.2f})".format(estimator, dict_evt[estimator]["rmse_bestK"],
                                                                        ), linestyle="-.")
    axes[0, 0].hlines(y=real_quantile, xmin=0., xmax=n_data,
                      label="reference line", color="black", linestyle="--")

    axes[0, 0].legend()
    axes[0, 0].spines["left"].set_color("black")
    axes[0, 0].spines["bottom"].set_color("black")

    # title / axis
    axes[0, 0].set_xlabel(r"anchor point $k$")
    axes[0, 0].set_ylabel("quantile")
    axes[0, 0].set_title("Bias estimator")

    axes[1, 0].set_xlabel(r"anchor point $k$")
    axes[1, 0].set_ylabel("variance")
    axes[1, 0].set_title("Variance estimator")
    axes[1, 0].spines["left"].set_color("black")
    axes[1, 0].spines["bottom"].set_color("black")

    axes[2, 0].set_xlabel(r"anchor point $k$")
    axes[2, 0].set_ylabel("RMSE")
    axes[2, 0].set_title("RMSE estimator")
    axes[2, 0].spines["left"].set_color("black")
    axes[2, 0].spines["bottom"].set_color("black")

    # y_lim
    # axes[0, 0].set_ylim(real_quantile*0.8, real_quantile*1.2)  # 100
    axes[0, 0].set_ylim(real_quantile*0.5, real_quantile*2) #real_quantile*3)  # QUANTILE
    axes[1, 0].set_ylim(np.min(dict_evt["CW"]["var"]) * 0.5, np.min(dict_evt["CW"]["var"]) * 2)  # VARIANCE
    # axes[1, 0].set_ylim(0, 22)  # VARIANCE
    axes[2, 0].set_ylim(0, 1)  # MSE

    fig.tight_layout()
    fig.suptitle("Estimator plot \n{}: {}".format(distribution.upper(), str(params).upper()), fontweight="bold", y=1.04)
    sns.despine()
    return


def evt_quantile_plot_paper(n_replications, n_data, distribution, params, n_quantile, plot_type, saved=False):
    """extreme quantile plot of just the evt estimatorsat level 1/2n for different replications with variance and MSE"""

    # LIST_ESTIMATORS_PAPER = ["W", "RW", "CW", "CHps", "PRBps"]
    LIST_ESTIMATORS_PAPER = ["W", "RW"]

    pathdir = Path("ckpt", n_quantile, distribution, "extrapolation", str(params))
    pathdir.mkdir(parents=True, exist_ok=True)

    anchor_points = np.arange(2, n_data)  # 1, ..., n-1
    if n_quantile == "2n":
        EXTREME_ALPHA = 1 / (2 * n_data)  # extreme alpha
    elif n_quantile == "n":
        EXTREME_ALPHA = 1 / (n_data)  # extreme alpha
    else:
        return "The 'n_quantile' doesn't exist. PLese choose between {'n', '2n'}."

    data_sampler = DataSampler(distribution=distribution, params=params)
    real_quantile = data_sampler.ht_dist.tail_ppf(1 / EXTREME_ALPHA)  # real extreme quantile

    try:
        dict_evt = np.load(Path(pathdir, "evt_estimators_rep{}.npy".format(n_replications)), allow_pickle=True)[()]
    except FileNotFoundError:
        dict_evt = evt_estimators(n_replications, n_data, distribution, params, n_quantile, return_full=True)

    fig, axes = plt.subplots(1, 1, figsize=(15, 7), sharex=False, squeeze=False)

    for estimator in LIST_ESTIMATORS_PAPER:
        if plot_type == "bias":
            axes[0, 0].plot(anchor_points, dict_evt[estimator]["series"], linestyle="-", linewidth=2)
            axes[0, 0].hlines(y=real_quantile, xmin=0., xmax=n_data, color="black", linestyle="--", linewidth=2)
        elif plot_type == "rmse":
            axes[0, 0].plot(anchor_points, dict_evt[estimator]["rmse"], linestyle="-", linewidth=2)

    axes[0, 0].spines["left"].set_color("black")
    axes[0, 0].spines["bottom"].set_color("black")

    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)

    if plot_type == "bias":
        axes[0, 0].set_ylim(real_quantile * 0.95, real_quantile * 1.2)  # real_quantile*3)  # QUANTILE
    elif plot_type == "rmse":
        axes[0, 0].set_ylim(0, .2)  # MSE


    fig.tight_layout()
    sns.despine()

    if saved:
        pathdir = Path("imgs")
        pathdir.mkdir(parents=True, exist_ok=True)
        filename = "{}-{}-{}-{}-{}-{}-".format(plot_type, distribution, params, n_replications, n_data,  n_quantile)
        plt.savefig(pathdir / "{}.eps".format(filename), format="eps")
    return


def evt_hill_plot(n_replications, n_data, distribution, params, n_quantile, saved=False):
    sns.set_style("whitegrid", {'grid.linestyle': '--'})

    pathdir = Path("ckpt", n_quantile, distribution, "extrapolation", str(params))
    pathdir.mkdir(parents=True, exist_ok=True)

    anchor_points = np.arange(2, n_data)  # 1, ..., n-1
    if n_quantile == "2n":
        EXTREME_ALPHA = 1 / (2 * n_data)  # extreme alpha
    elif n_quantile == "n":
        EXTREME_ALPHA = 1 / (n_data)  # extreme alpha
    else:
        return "The 'n_quantile' doesn't exist. PLese choose between {'n', '2n'}."

    data_sampler = DataSampler(distribution=distribution, params=params)
    X_order = data_sampler.simulate_quantiles(n_data, seed=1)  # new quantiles X_1,n, ..., X_n,n

    fig, axes = plt.subplots(1, 1, figsize=(15, 7), sharex=False, squeeze=False)  # 3 plots: quantile, var, mse

    evt_estimators = ExtremeQuantileEstimator(X=X_order, alpha=EXTREME_ALPHA)
    anchor_points = np.arange(2, n_data)  # 2, ..., n-1


    hill_gammas = [evt_estimators.hill(k_anchor) for k_anchor in anchor_points]
    bestK = random_forest_k(np.array(hill_gammas), n_forests=10000, seed=42)

    k_prime = evt_estimators.get_kprime_rw(n_data-1)[0]
    print(k_prime)
    anchor_points_prime = np.arange(2, int(k_prime)+1)
    hill_gammas_prime = [evt_estimators.hill(k_anchor) for k_anchor in anchor_points_prime]

    axes[0, 0].plot(anchor_points, hill_gammas, color="black")
    # axes[0, 0].scatter(bestK , hill_gammas[bestK], s=200, color="red", marker="^")
    axes[0, 0].plot(anchor_points_prime, hill_gammas_prime, color="red")
    axes[0, 0].hlines(y=params["evi"], xmin=0., xmax=n_data, color="black", linestyle="--")

    axes[0, 0].spines["left"].set_color("black")
    axes[0, 0].spines["bottom"].set_color("black")

    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)

    # y_lim
    # axes[0, 0].set_ylim(params["evi"] * 0.4, params["evi"]  * 2.5)  # 100
    axes[0, 0].set_ylim(params["evi"] * 0.4, params["evi"]  * 2.5)  # 100

    fig.tight_layout()
    sns.despine()

    if saved:
        pathdir = Path("imgs")
        pathdir.mkdir(parents=True, exist_ok=True)
        plt.savefig(pathdir / "hill_plot_evt.eps", format="eps")
        filename = "hill-{}-{}-{}-{}-{}-".format(distribution, params, n_replications, n_data,  n_quantile)
        plt.savefig(pathdir / "{}.eps".format(filename), format="eps")
    return

# ====================================================================================================================
# Real plot
# ---------

def real_quantile_plot(saved=False):
    sns.set_style("whitegrid", {'grid.linestyle': '--'})
    fig, axes = plt.subplots(1, 1, figsize=(15, 7), sharex=False, squeeze=False)  # 3 plots: quantile, var, mse

    X = pd.read_csv(Path(os.getcwd(), 'dataset', "besecura.txt"), sep='\t').loc[:, 'Loss'].to_numpy()  # read data
    X_order = np.sort(X)
    n_data = len(X_order)
    anchor_points = np.arange(2, n_data)  # 2, ..., n-1
    real_quantile = X_order[-1]  # real extreme quantile at order 1/n

    dict_evt = real_estimators(return_full=True)

    # plot EVT estimator
    for estimator in list_estimators:  # list_estimators (all estimators)
        lab ="{} (k={}, q={:.2f})".format(estimator, int(dict_evt[estimator]["bestK"][0]), np.array(dict_evt[estimator]["q_bestK"]).ravel()[0])
        axes[0, 0].plot(anchor_points, np.array(dict_evt[estimator]["series"]).ravel(), label=lab)
        axes[0, 0].scatter(dict_evt[estimator]["bestK"], dict_evt[estimator]["q_bestK"], s=100)


    # plot reference line
    axes[0, 0].hlines(y=real_quantile, xmin=0., xmax=n_data, color="black", linestyle="--")
    # label="reference line (q={:.2f})".format(float(real_quantile))

    # axes[0, 0].legend()
    axes[0, 0].spines["left"].set_color("black")
    axes[0, 0].spines["bottom"].set_color("black")

    # y_lim
    axes[0, 0].set_ylim(real_quantile * 0.7, real_quantile * 1.6)  # 100

    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    axes[0, 0].yaxis.offsetText.set_fontsize(18)

    fig.tight_layout()
    sns.despine()
    if saved:
        pathdir = Path("imgs")
        pathdir.mkdir(parents=True, exist_ok=True)
        plt.savefig(pathdir / "quantile_plot_real.eps", format="eps")
    return

def real_quantile_plot_paper(saved=False):
    sns.set_style("whitegrid", {'grid.linestyle': '--'})
    fig, axes = plt.subplots(1, 1, figsize=(15, 7), sharex=False, squeeze=False)  # 3 plots: quantile, var, mse

    X = pd.read_csv(Path(os.getcwd(), 'dataset', "besecura.txt"), sep='\t').loc[:, 'Loss'].to_numpy()  # read data
    X_order = np.sort(X)
    n_data = len(X_order)
    anchor_points = np.arange(2, n_data)  # 2, ..., n-1
    real_quantile = X_order[-1]  # real extreme quantile at order 1/n

    dict_evt = real_estimators(return_full=True)

    axes[0, 0].plot(anchor_points, np.array(dict_evt["RW"]["series"]).ravel(), color="C1")
    axes[0, 0].scatter(dict_evt["RW"]["bestK"], dict_evt["RW"]["q_bestK"], s=200, marker="^", color="C1")

    axes[0, 0].plot(anchor_points, np.array(dict_evt["CW"]["series"]).ravel(), color="C2")
    axes[0, 0].scatter(dict_evt["CW"]["bestK"], dict_evt["CW"]["q_bestK"], s=200,marker="^", color="C2")


    # plot reference line
    axes[0, 0].hlines(y=real_quantile, xmin=0., xmax=n_data, color="black", linestyle="--")
    # label="reference line (q={:.2f})".format(float(real_quantile))

    # axes[0, 0].legend()
    axes[0, 0].spines["left"].set_color("black")
    axes[0, 0].spines["bottom"].set_color("black")

    # y_lim
    axes[0, 0].set_ylim(real_quantile * 0.7, real_quantile * 1.6)  # 100

    plt.xticks(fontsize=20)
    plt.yticks(np.arange(0.6, 1.3, 0.1)*1e7, labels=[0.6, 0.7,0.8,0.9,1., 1.1, 1.2], fontsize=20)


    fig.tight_layout()
    sns.despine()
    if saved:
        pathdir = Path("imgs")
        pathdir.mkdir(parents=True, exist_ok=True)
        plt.savefig(pathdir / "quantile_plot_real.eps", format="eps")
    return

def real_hill_plot(saved=False):
    sns.set_style("whitegrid", {'grid.linestyle': '--'})
    fig, axes = plt.subplots(1, 1, figsize=(15, 7), sharex=False, squeeze=False)  # 3 plots: quantile, var, mse

    X = pd.read_csv(Path(os.getcwd(), 'dataset', "besecura.txt"), sep='\t').loc[:, 'Loss'].to_numpy()  # read data
    X_order = np.sort(X)
    n_data = len(X_order)
    anchor_points = np.arange(2, n_data)  # 2, ..., n-1
    anchor_points_prime = np.arange(2, 107)
    EXTREME_ALPHA = 1 / n_data
    evt_estimators = ExtremeQuantileEstimator(X=X_order, alpha=EXTREME_ALPHA)

    K_STAR = 68  # k^star chosen

    hill_gammas = [evt_estimators.hill(k_anchor) for k_anchor in anchor_points]
    hill_gammas_prime = [evt_estimators.hill(k_anchor) for k_anchor in anchor_points_prime]

    axes[0, 0].plot(anchor_points, hill_gammas, color="black")
    axes[0, 0].scatter(K_STAR , hill_gammas[K_STAR -1], s=200, color="red", marker="^")
    axes[0, 0].plot(anchor_points_prime, hill_gammas_prime, color="red")


    axes[0, 0].spines["left"].set_color("black")
    axes[0, 0].spines["bottom"].set_color("black")

    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)



    fig.tight_layout()
    sns.despine()

    if saved:
        pathdir = Path("imgs")
        pathdir.mkdir(parents=True, exist_ok=True)
        plt.savefig(pathdir / "hill_plot_real.eps", format="eps")
    return

def real_loglog_plot(saved=False):
    sns.set_style("whitegrid", {'grid.linestyle': '--'})
    fig, axes = plt.subplots(1, 1, figsize=(15, 7), sharex=False, squeeze=False)  # 3 plots: quantile, var, mse

    X = pd.read_csv(Path(os.getcwd(), 'dataset', "besecura.txt"), sep='\t').loc[:, 'Loss'].to_numpy()  # read data
    X_order = np.sort(X)
    n_data = len(X_order)
    K_STAR = 68
    anchor_points = np.arange(2, n_data)  # 2, ..., n-1
    i_points = np.arange(1, K_STAR)
    y = np.log(X_order[-i_points]) - np.log(X_order[-K_STAR])
    X = np.log(K_STAR /  i_points)
    EXTREME_ALPHA = 1 / n_data
    evt_estimators = ExtremeQuantileEstimator(X=X_order, alpha=EXTREME_ALPHA)

    hill_gammas = [evt_estimators.hill(k_anchor) for k_anchor in anchor_points]
    gamma = hill_gammas[K_STAR -1]

    axes[0, 0].scatter(X, y, s=100, color="black", marker="+")
    axes[0, 0].plot(X, X * gamma, color="red")

    axes[0, 0].spines["left"].set_color("black")
    axes[0, 0].spines["bottom"].set_color("black")

    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)

    fig.tight_layout()
    sns.despine()
    if saved:
        pathdir = Path("imgs")
        pathdir.mkdir(parents=True, exist_ok=True)
        plt.savefig(pathdir / "loglog_plot_real.eps", format="eps")
    return


def real_hist_plot(saved=False):
    # sns.set_style("whitegrid", {'grid.linestyle': '--'})
    # fig, axes = plt.subplots(1, 1, figsize=(15, 7), sharex=False, squeeze=False)  # 3 plots: quantile, var, mse
    # plt.figure(figsize=(20, 12))

    X = pd.read_csv(Path(os.getcwd(), 'dataset', "besecura.txt"), sep='\t').loc[:, 'Loss'].to_numpy()  # read data

    h = sns.displot(data=X, aspect=2, height=10)
    # sns.histplot(data=X)
    h.set(ylabel=None)  # remove the axis label
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)

    h.set(xticks=[1e6, 2e6, 3e6, 4e6, 5e6, 6e6, 7e6, 8e6])
    h.set_xticklabels(np.arange(1, 9, 1))

    sns.despine()
    if saved:
        pathdir = Path("imgs")
        pathdir.mkdir(parents=True, exist_ok=True)
        plt.savefig(pathdir / "hist_real.eps", format="eps")
    return

