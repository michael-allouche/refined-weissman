import numpy as np
#import torch
import matplotlib.pyplot as plt
import seaborn as sns
from IPython import display
import time

from utils import load_summary_file
from extreme.estimators import  evt_estimators, random_forest_k
from extreme.data_management import DataSampler
# from models import load_model, get_best_crit
import re
from pathlib import Path

# from rpy2 import robjects as ro
# import rpy2.robjects.numpy2ri




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
    axes[0, 0].set_title("Extreme quantile estimator")

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
    fig.suptitle("Estimator plot \n{}: {}".format(distribution.capitalize(), str(params).upper()), fontweight="bold", y=1.04)
    sns.despine()
    return


def plot_bestK(epoch=None, show_as_video=False, **model_filenames):
    # take the first one to load the config summary and extract data infos
    sns.set_style("whitegrid", {'grid.linestyle': '--'})
    colors = plt.cm.rainbow(np.linspace(0, 1, len(model_filenames)))

    _, model_filename = list(model_filenames.items())[0]
    summary_file = load_summary_file(model_filename)
    seed = int("".join(re.findall('rep([0-9]+)*$', model_filename)))

    n_data = summary_file["n_data"]

    anchor_points = np.arange(2, n_data)  # 2, ..., n-1
    EXTREME_ALPHA = 1 / (2 * n_data)  # extreme alpha

    # real data
    data_sampler = DataSampler(**summary_file)
    real_quantile = data_sampler.ht_dist.tail_ppf(1 / EXTREME_ALPHA)  # real extreme quantile
    X_order = data_sampler.simulate_quantiles(n_data, seed=seed)  # new order statistics X_1,n, ..., X_n,n
    # evt_estimators = ExtremeQuantileEstimator(X=X_order, alpha=EXTREME_ALPHA)

    if epoch is None:  # by defaut the epoch selected is the last one
        epoch = summary_file["n_epochs"]

    def _xquantile_video(epoch):
        fig, axes = plt.subplots(1, 1, figsize=(15, 7), sharex=False, squeeze=False)  # 3 plots: quantile, var, mse

        for idx_model, (order_condition, model_filename) in enumerate(model_filenames.items()):
            # model quantiles
            model = load_model(model_filename, epoch, summary_file["distribution"])
            # X_order in reverse order without X_1,n
            q_nn = model.extrapolate(alpha=EXTREME_ALPHA, k_anchor=anchor_points, X_order=X_order).ravel()
            # bestK_nn = dyadic_k(np.array(q_nn))  # i=0, ... , 497
            bestK_nn = random_forest_k(np.array(q_nn), 10000)  # i=0, ... , 497
            bestK_nn_100 = random_forest_k(np.array(q_nn), 100)
            bestK_nn_1000 = random_forest_k(np.array(q_nn), 1000)
            bestK_nn_10000 = random_forest_k(np.array(q_nn), 10000)

            # plot NN estimator
            axes[0, 0].plot(anchor_points, q_nn, label="{}".format(order_condition), color=colors[idx_model])
            axes[0, 0].scatter(int(bestK_nn), q_nn[int(bestK_nn)], s=100,  label="no random: (k={}, q={:.2f})".format(int(bestK_nn) + 2,
                                                                                     q_nn[int(bestK_nn)]))
            axes[0, 0].scatter(int(bestK_nn_100), q_nn[int(bestK_nn_100)], s=100,  label="RF100: (k={}, q={:.2f})".format(int(bestK_nn_100) + 2,
                                                                                     q_nn[int(bestK_nn_100)]))
            axes[0, 0].scatter(int(bestK_nn_1000), q_nn[int(bestK_nn_1000)], s=100,  label="RF1000: (k={}, q={:.2f})".format(int(bestK_nn_1000) + 2,
                                                                                     q_nn[int(bestK_nn_1000)]))
            axes[0, 0].scatter(int(bestK_nn_10000), q_nn[int(bestK_nn_10000)], s=100,  label="RF10000: (k={}, q={:.2f})".format(int(bestK_nn_10000) + 2,
                                                                                     q_nn[int(bestK_nn_10000)]))


        # plot reference line

        axes[0, 0].hlines(y=real_quantile, xmin=0., xmax=n_data,
                          label="reference line (q={:.2f})".format(float(real_quantile)), color="black", linestyle="--")

        axes[0, 0].legend()
        axes[0, 0].spines["left"].set_color("black")
        axes[0, 0].spines["bottom"].set_color("black")

        # title / axis
        axes[0, 0].set_xlabel(r"anchor point $k$")
        axes[0, 0].set_ylabel("quantile")

        # y_lim
        axes[0, 0].set_ylim(real_quantile * -0.5, real_quantile * 1.5)  # 100

        fig.tight_layout()
        fig.suptitle("Estimator plot\n{}: {} \n(epoch={})".format(summary_file["distribution"].capitalize(),
                                                                  str(summary_file["params"]).upper(),
                                                                  epoch), fontweight="bold", y=1.04)
        sns.despine()
        return

    if show_as_video:
        save_freq = summary_file["verbose"]
        ckpt_epochs = [save_freq] + [i for i in range(save_freq, epoch + save_freq, save_freq)]
        for chkpt_epoch in ckpt_epochs:
            _xquantile_video(chkpt_epoch)
            plt.show()
            display.clear_output(wait=True)
            # time.sleep(1)
    else:
        _xquantile_video(epoch)

    return





