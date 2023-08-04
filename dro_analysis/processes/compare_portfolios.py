import os
from dro_analysis import paths
from dro_analysis.utility_functions.df_manipulation import calendar_year_returns
from dro_analysis.utility_functions.utils import save_pickle, load_pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import riskfolio as rp


if __name__ == "__main__":
    # get prices
    data_file = 'PublicData'
    prices = pd.read_pickle(os.path.join(paths.CLEAN_DATA, data_file))
    prices = prices.loc[prices.index > pd.Timestamp(1990, 1, 30)]  # no need to cut, already done in functions
    prices = prices.drop('TIPS US', axis=1)

    # get yearly returns
    yearly_returns = calendar_year_returns(prices)
    yearly_returns = yearly_returns.drop(yearly_returns.index[0])

    # load generated portfolios and corresponding performance
    # portfolios = load_pickle(os.path.join(paths.PORTFOLIOS, 'portfolios1'))
    # performance = load_pickle(os.path.join(paths.PORTFOLIOS, 'performance1'))
    # values = [10, 15, 20, 25, 30, 35, 40, 45, 50]
    portfolios = load_pickle(os.path.join(paths.PORTFOLIOS, 'testos_portfolios'))
    performance = load_pickle(os.path.join(paths.PORTFOLIOS, 'testosperformance'))
    values = [10, 50]

    # plot
    fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(12, 9))
    for col, i in enumerate(performance):
        for row, j in enumerate(performance[i]):
            x_data = []
            y_data = []
            for strat, point in performance[i][j].items():
                # axes[row, col].scatter(point[0], point[1], c=colors[strat])
                if strat == 'standard':
                    axes[row, col].scatter(point[0], point[1], c='red', label=f'classic {i}')
                else:
                    x_data.append(point[0])
                    y_data.append(point[1])
            axes[row, col].scatter(x_data, y_data, c=values, cmap='Blues', label=f'robust {i}, incr. radius')
            axes[row, col].axhline(y=(1.1**22-1)*100, color='black', linestyle='--', label='Return constraint')
            axes[row, col].set_title(f'{i} {j}')
            axes[row, col].legend()

    plt.tight_layout()
    # fig.text(0.5, 0.04, 'variance', ha='center')
    # fig.text(0.04, 0.5, 'return %', va='center', rotation='vertical')
    # plt.suptitle("Empirical return and variance between 2001 and 2023 (22 years)")
    plt.show()





