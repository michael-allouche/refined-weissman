import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from IPython import display

from utils import load_summary_file
from extreme.estimators import  evt_estimators, random_forest_k
from extreme.data_management import DataSampler
import re
from pathlib import Path





def evt_quantile_plot(n_replications, n_data, distribution, params, metric):
    """extreme quantile plot of just the evt estimatorsat level 1/2n for different replications with variance and MSE"""

    pathdir = Path("ckpt", distribution, "extrapolation", str(params))
    pathdir.mkdir(parents=True, exist_ok=True)

    anchor_points = np.arange(2, n_data)  # 1, ..., n-1
    EXTREME_ALPHA = 1 / (2 * n_data)  # extreme alpha
    data_sampler = DataSampler(distribution=distribution, params=params)
    real_quantile = data_sampler.ht_dist.tail_ppf(1 / EXTREME_ALPHA)  # real extreme quantile

    try:
        dict_evt = np.load(Path(pathdir, "evt_estimators_rep{}.npy".format(n_replications)), allow_pickle=True)[()]
    except FileNotFoundError:
        dict_evt = evt_estimators(n_replications, n_data, distribution, params, return_full=True)

    fig, axes = plt.subplots(3, 1, figsize=(15, 3 * 5), sharex=False, squeeze=False)  # 3 plots: quantile, var, mse

    for estimator in dict_evt.keys():
        axes[0, 0].plot(anchor_points, dict_evt[estimator][metric]["series"],
                        label="{} (rmse={:.2f})".format(estimator, dict_evt[estimator][metric]["rmse_bestK"],
                                                                        ), linestyle="-.")
        axes[1, 0].plot(anchor_points, dict_evt[estimator][metric]["var"],
                        label="{} (rmse={:.2f})".format(estimator, dict_evt[estimator][metric]["rmse_bestK"],
                                                                        ), linestyle="-.")
        axes[2, 0].plot(anchor_points, dict_evt[estimator][metric]["rmse"],
                        label="{} (rmse={:.2f})".format(estimator, dict_evt[estimator][metric]["rmse_bestK"],
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
    axes[0, 0].set_ylim(real_quantile*0.5, real_quantile*3)  # QUANTILE
    axes[1, 0].set_ylim(np.min(dict_evt["W"][metric]["var"]) * 0.5, np.min(dict_evt["W"][metric]["var"]) * 10)  # VARIANCE
    # axes[1, 0].set_ylim(0, 22)  # VARIANCE
    axes[2, 0].set_ylim(0, 1)  # MSE

    fig.tight_layout()
    fig.suptitle("Estimator plot \n{}: {}".format(distribution.upper(), str(params).upper()), fontweight="bold", y=1.04)
    sns.despine()
    return






