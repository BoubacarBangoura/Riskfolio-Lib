import os
from experiments import paths
from experiments.utility_functions.df_manipulation import calendar_year_returns, mu_cov_prev_years, mu_mean_reversion, yearly_perf, total_perf
from experiments.utility_functions.utils import save_pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import riskfolio as rp


def compute_portfolios(returns, test_start, strategy, strategy_rm, estimation_method, radius=0, lowerret=0.1):

    # create empty dataframe to fill up with portfolio weights
    results = returns.loc[returns.index > test_start]
    results = results.apply(lambda x: pd.Series([None] * len(x), index=x.index))

    # visit all test times
    test_times = returns.index[returns.index >= test_start]
    for t in test_times:
        # estimate mu and cov
        if estimation_method in ['5_prev_years', 'mean_rev_5_years']:
            years = 5
        else:
            years = 10
        cut_returns = returns[returns.index < t]
        cut_returns = cut_returns[-int(years):]
        if estimation_method in ['5_prev_years', '10_prev_years']:
            mu, cov = mu_cov_prev_years(returns, t, years)
        else:
            mu, cov = mu_mean_reversion(returns, t, years)

        # create portfolio
        p = rp.Portfolio(returns=cut_returns)
        p.mu = mu
        p.cov = cov
        if strategy == 'Blanchet':
            p.lowerrobret = lowerret
        elif strategy in ['MV', 'CVaR', 'robVaR', 'robCVaR']:
            p.lowerret = lowerret

        # optimize portfolio
        if strategy == 'utility':
            obj = 'Utility'
        else:
            obj = 'MinRisk'
        p.optimization(model='Classic', rm=strategy_rm[strategy], obj=obj, rf=0, l=2, hist=True, radius=radius)
        results.loc[t] = p.optimal['weights']
    return results


if __name__ == "__main__":
    # get prices
    data_file = 'PublicData'
    prices = pd.read_pickle(os.path.join(paths.CLEAN_DATA, data_file))
    prices = prices.loc[prices.index > pd.Timestamp(1990, 1, 30)]  # no need to cut, already done in functions
    prices = prices.drop('TIPS US', axis=1)

    # get monthly and yearly returns
    monthly_returns = prices.pct_change().dropna()
    yearly_returns = calendar_year_returns(prices)
    yearly_returns = yearly_returns.drop(yearly_returns.index[0])

    # labels and parameters
    strategies = ['MV', 'utility', 'CVaR', 'Blanchet', 'robutility', 'robVaR', 'robCVaR']
    strategies_rm = {'MV': 'MV', 'utility': 'MV', 'CVaR': 'CVaR', 'Blanchet': 'robvariance', 'robutility': 'robmeandev', 'robVaR': 'robVaR', 'robCVaR': 'robCVaR'}
    rob_strategies = ['Blanchet', 'robutility', 'robVaR', 'robCVaR']
    parameters_estimation = ['5_prev_years', '10_prev_years', 'mean_rev_5_years']
    radii = [0.01, 0.05, 0.1, 0.5, 1, 5, 10, 50, 100]
    _test_start = pd.to_datetime('2001-12-31')
    _lowerret = 0.1

    # we create an empty dictionary of dictionaries where we store the portfolios for different strategies
    portfolios = {index: None for index in strategies}
    for index in portfolios:
        portfolios[index] = {i: None for i in parameters_estimation}
        if index in rob_strategies:
            for i in portfolios[index]:
                portfolios[index][i] = {j: None for j in radii}

    # best strategies
    performance = {}

    # fill up the portfolios dictionary
    iteration = 1
    for _strategy in strategies:
        for _estimation_method in parameters_estimation:
            if _strategy in rob_strategies:
                for _radius in radii:
                    print(f'---------- iteration {iteration} ----------')
                    iteration += 1
                    result = compute_portfolios(returns=yearly_returns, test_start=_test_start, strategy=_strategy, strategy_rm=strategies_rm, estimation_method=_estimation_method, radius=_radius, lowerret=_lowerret)
                    portfolios[_strategy][_estimation_method][_radius] = result
                    performance[(_strategy, _estimation_method, _radius)] = (total_perf(portfolio=result, returns=yearly_returns), yearly_perf(result, yearly_returns).var())
            else:
                print(f'---------- iteration {iteration} ----------')
                iteration += 1
                result = compute_portfolios(returns=yearly_returns, test_start=_test_start, strategy=_strategy,
                                            strategy_rm=strategies_rm, estimation_method=_estimation_method,
                                            lowerret=_lowerret)
                portfolios[_strategy][_estimation_method] = result
                performance[(_strategy, _estimation_method)] = (total_perf(result, yearly_returns), yearly_perf(result, yearly_returns).var())

    save_pickle(portfolios, os.path.join(paths.PORTFOLIOS, 'portfolios1'))
    save_pickle(performance, os.path.join(paths.PORTFOLIOS, 'performance1'))

