import os
from dro_analysis import paths

import matplotlib.pyplot as plt
from matplotlib import cm, colors
import numpy as np
import pandas as pd
import riskfolio as rp


if __name__ == "__main__":
    # run parameters:
    time_delta = 'M'  # Y or M
    save = False
    radius = 4 * 10 ** (-6)
    realization = True

    # get prices
    data_file = 'PublicData'
    prices = pd.read_pickle(os.path.join(paths.CLEAN_DATA, data_file))
    prices = prices.loc[prices.index > pd.Timestamp(1990, 1, 30)]  # no need to cut, already done in functions
    prices = prices.drop('TIPS US', axis=1)

    t_factor = {'M': 12, 'Y': 1}[time_delta]
    returns = prices.resample(time_delta).last().pct_change().dropna()
    if time_delta == 'Y':
        test_size = 10
        test_returns = returns.iloc[-test_size:]
        returns = returns.iloc[:-test_size]
    elif time_delta == 'M':
        returns = returns.iloc[:-48]
        test_size = 18
        test_returns = returns.iloc[-test_size:]
        returns = returns.iloc[:-test_size]

    # prices.resample('Y').last().pct_change().rolling(10).mean()
    # prices.resample('Y').last().pct_change().rolling(10).mean().plot()
    # prices.resample('Y').last().pct_change().rolling(10).std().plot()

    # parameters
    method_mu, method_cov, model, hist = 'hist', 'hist', 'Classic', True
    lowerrobret = 0.1 / t_factor
    lowerret = 0.06 / t_factor

    port = rp.Portfolio(returns=returns)
    port.assets_stats(method_mu=method_mu, method_cov=method_cov, d=0.94)

    # generate portfolios for the efficient frontiers of MV and robust
    points = 50  # Number of points of the frontier
    mu = port.mu  # Expected returns
    cov = port.cov  # Covariance matrix
    label = 'Max Risk Adjusted Return Portfolio'  # Title of point

    rm = "MV"
    kelly = False
    frontier = port.efficient_frontier(points=points, radius=radius)
    frontier_rob = port.efficient_frontier(rm='robvariance', kelly='robmean', points=points, radius=radius)

    # plot efficient frontier
    ax = rp.plot_robust_frontier(w_frontier=frontier, w_frontier_rob=frontier_rob, mu=mu, cov=cov, returns=returns,
           test_returns=test_returns, t_factor=t_factor, ax=None, radius=radius, realized=realization)
    if realization:
        name = f'ef_real.png'
    else:
        name = f'ef.png'
    if save:
        plt.savefig(os.path.join(f'radius_-{radius}', name), dpi=300)
    plt.show()

    # plot composition of MV
    ax = rp.plot_frontier_area(w_frontier=frontier, cmap="tab20", height=6, width=10, ax=None)
    if save:
        plt.savefig(os.path.join(f'radius_-{radius}', 'area_mv_full.png'), dpi=300)
    plt.show()

    # plot composition of robust
    ax = rp.plot_frontier_area(w_frontier=frontier_rob, cmap="tab20", height=6, width=10, ax=None)
    if save:
        plt.savefig(os.path.join(f'radius_-{radius}', 'area_rob_full.png'), dpi=300)
    plt.show()

    # fig, axs = plt.subplots(1, 2)
    # nrow = 25
    # cmap = "tab20"
    # n_colors = 20
    #
    # # fig 1
    # w_frontier = frontier
    #
    # axs[0].set_title("Efficient Frontier's Assets Structure")
    # labels = w_frontier.index.tolist()
    #
    # colormap = cm.get_cmap(cmap)
    # colormap = colormap(np.linspace(0, 1, n_colors))
    #
    # if cmap == "gist_rainbow":
    #     colormap = colormap[::-1]
    #
    # cycle = plt.cycler("color", colormap)
    # axs[0].set_prop_cycle(cycle)
    #
    # X = w_frontier.columns.tolist()
    #
    # axs[0].stackplot(X, w_frontier, labels=labels, alpha=0.7, edgecolor="black")
    #
    # axs[0].set_ylim(0, 1)
    # axs[0].set_xlim(0, len(X) - 1)
    #
    # ticks_loc = ax.get_yticks().tolist()
    # axs[0].set_yticks(ax.get_yticks().tolist())
    # axs[0].set_yticklabels(["{:3.2%}".format(x) for x in ticks_loc])
    # axs[0].grid(linestyle=":")
    #
    # n = int(np.ceil(len(labels) / nrow))
    #
    # # fig 2
    # w_frontier = frontier_rob
    #
    # axs[1].set_title("Efficient Frontier's Assets Structure")
    # labels = w_frontier.index.tolist()
    #
    # colormap = cm.get_cmap(cmap)
    # colormap = colormap(np.linspace(0, 1, n_colors))
    #
    # if cmap == "gist_rainbow":
    #     colormap = colormap[::-1]
    #
    # cycle = plt.cycler("color", colormap)
    # axs[1].set_prop_cycle(cycle)
    #
    # X = w_frontier.columns.tolist()
    #
    # axs[1].stackplot(X, w_frontier, labels=labels, alpha=0.7, edgecolor="black")
    #
    # axs[1].set_ylim(0, 1)
    # axs[1].set_xlim(0, len(X) - 1)
    #
    # ticks_loc = ax.get_yticks().tolist()
    # axs[1].set_yticks(ax.get_yticks().tolist())
    # axs[1].set_yticklabels(["{:3.2%}".format(x) for x in ticks_loc])
    # axs[1].grid(linestyle=":")
    #
    # n = int(np.ceil(len(labels) / nrow))
    #
    # axs[1].legend(labels, loc="center left", bbox_to_anchor=(1, 0.5), ncol=n)
    #
    # fig.tight_layout()
    # plt.show()

