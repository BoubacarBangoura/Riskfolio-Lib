import os
from experiments import paths
from experiments.utility_functions.df_manipulation import calendar_year_returns, mu_cov_prev_years, mu_mean_reversion, yearly_perf,  total_perf
from experiments.utility_functions.utils import save_pickle, load_pickle

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

    portfolios = load_pickle(os.path.join(paths.PORTFOLIOS, 'portfolios1'))
    # performance = load_pickle(os.path.join(paths.PORTFOLIOS, 'performance1'))

    # labels and parameters
    strategies = ['MV', 'utility', 'CVaR', 'Blanchet', 'robutility', 'robVaR', 'robCVaR']
    strategies_rm = {'MV': 'MV', 'utility': 'MV', 'CVaR': 'CVaR', 'Blanchet': 'robvariance', 'robutility': 'robmeandev', 'robVaR': 'robVaR', 'robCVaR': 'robCVaR'}
    rob_strategies = ['Blanchet', 'robutility', 'robVaR', 'robCVaR']
    parameters_estimation = ['5_prev_years', '10_prev_years', 'mean_rev_5_years']
    radii = [0.01, 0.05, 0.1, 0.5, 1, 5, 10, 50, 100]
    _test_start = pd.to_datetime('2001-12-31')

    # ...
    performance = {i: None for i in ['MV', 'utility', 'CVaR']}
    for i in performance:
        performance[i] = {j: None for j in ['5_prev_years', '10_prev_years', 'mean_rev_5_years']}
        for j in performance[i]:
            performance[i][j] = {k: None for k in ['standard', 'radius 0.01', 'radius 0.05', 'radius 0.1', 'radius 0.5', 'radius 1', 'radius 5', 'radius 10', 'radius 50', 'radius 100']}
            for k in performance[i][j]:
                if i == 'MV':
                    if k == 'standard':
                        result = portfolios['MV'][j]
                        print(result.index.size)
                    else:
                        result = portfolios['Blanchet'][j][float(k[7:])]
                if i == 'utility':
                    if k == 'standard':
                        result = portfolios['utility'][j]
                    else:
                        result = portfolios['robutility'][j][float(k[7:])]
                if i == 'CVaR':
                    if k == 'standard':
                        result = portfolios['CVaR'][j]
                    else:
                        result = portfolios['robCVaR'][j][float(k[7:])]
                performance[i][j][k] = (yearly_perf(result, yearly_returns).var(), total_perf(portfolio=result, returns=yearly_returns))

    save_pickle(performance, os.path.join(paths.PORTFOLIOS, 'performance1'))

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
            values = [10, 15, 20, 25, 30, 35, 40, 45, 50]
            axes[row, col].scatter(x_data, y_data, c=values, cmap='Blues', label=f'robust {i}, incr. radius')
            axes[row, col].axhline(y=(1.1**22-1)*100, color='black', linestyle='--', label='Return constraint')
            axes[row, col].set_title(f'{i} {j}')
            axes[row, col].legend()

    plt.tight_layout()
    # fig.text(0.5, 0.04, 'variance', ha='center')
    # fig.text(0.04, 0.5, 'return %', va='center', rotation='vertical')
    # plt.suptitle("Empirical return and variance between 2001 and 2023 (22 years)")
    plt.show()





