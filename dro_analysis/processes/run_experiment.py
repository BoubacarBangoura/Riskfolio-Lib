import os
from dro_analysis import paths
from dro_analysis.utility_functions.df_manipulation import calendar_year_returns, compute_mu_cov, yearly_perf, total_perf
from dro_analysis.utility_functions.utils import save_pickle
from dro_analysis.constants import MV, robMV, utility, robutility, CVaR, robCVaR, robVaR
from dro_analysis.constants import prev_5_years, prev_10_years, mean_rev_5_years
from dro_analysis.financial_objects.Portfolios import Portfolios

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import riskfolio as rp


def compute_portfolios(returns, test_start, strategy):

    # create empty dataframe to fill up with portfolio weights
    results = returns.loc[returns.index > test_start]
    results = results.apply(lambda x: pd.Series([None] * len(x), index=x.index))

    # visit all test times
    test_times = returns.index[returns.index >= test_start]
    for t in test_times:
        years = strategy.estimation_method['years']
        cut_returns = returns[returns.index < t]
        cut_returns = cut_returns[-int(years):]

        # create portfolio
        p = rp.Portfolio(returns=cut_returns)
        p.mu, p.cov = compute_mu_cov(returns, t, **strategy.estimation_method)
        p.lowerrobret = strategy.port_parameters.get('lowerrobret', None)
        p.lowerret = strategy.port_parameters.get('lowerret', None)
        p.optimization(**strategy.opt_parameters)

        results.loc[t] = p.optimal['weights']

    return results


if __name__ == "__main__":
    experiment_name = 'testos'

    # get prices
    data_file = 'PublicData'
    prices = pd.read_pickle(os.path.join(paths.CLEAN_DATA, data_file))
    prices = prices.loc[prices.index > pd.Timestamp(1990, 1, 30)]  # no need to cut, already done in functions
    prices = prices.drop('TIPS US', axis=1)

    # get monthly and yearly returns
    monthly_returns = prices.pct_change().dropna()
    yearly_returns = prices.resample('Y').last().pct_change()

    strategies = {'MV': MV, 'robMV': robMV, 'utility': utility, 'robutility': robutility, 'CVaR': CVaR, 'robCVaR': robCVaR, 'robVaR': robVaR}
    estimation_methods = {'prev_5_years': prev_5_years, 'prev_10_years': prev_10_years, 'mean_rev_5_years': mean_rev_5_years}

    radii = [0.01, 1]
    _test_start = pd.to_datetime('2001-12-31')

    # we create an empty dictionary of dictionaries where we store the portfolios for different strategies
    portfolios = {index: None for index in strategies}
    for index in portfolios:
        portfolios[index] = {i: None for i in estimation_methods}
        if index in ['robMV', 'robutility', 'robCVaR', 'robVaR']:
            for i in portfolios[index]:
                portfolios[index][i] = {j: None for j in radii}

    # fill up the portfolios dictionary
    iteration = 1
    for _s in strategies:
        _strategy = strategies[_s]
        for _estimation_method in estimation_methods:
            _strategy.estimation_method = estimation_methods[_estimation_method]
            if _s in ['robMV', 'robutility', 'robCVaR', 'robVaR']:
                for _radius in radii:
                    print(f'---------- iteration {iteration} ----------')
                    iteration += 1
                    _strategy.opt_parameters['radius'] = _radius
                    result = compute_portfolios(returns=yearly_returns, test_start=_test_start, strategy=_strategy)
                    portfolios[_s][_estimation_method][_radius] = Portfolios(result, datafile=data_file, test_start=_test_start, strategy=_strategy, name=f'p{iteration}')
            else:
                print(f'---------- iteration {iteration} ----------')
                iteration += 1
                result = compute_portfolios(returns=yearly_returns, test_start=_test_start, strategy=_strategy)
                portfolios[_s][_estimation_method] = Portfolios(result, datafile=data_file, test_start=_test_start, strategy=_strategy, name=f'p{iteration}')
    save_pickle(portfolios, os.path.join(paths.PORTFOLIOS, experiment_name+'_portfolios'))

    # get performance (actual return and variance) of the portfolios
    performance = {i: None for i in ['MV', 'utility', 'CVaR']}
    radii_indices = ['standard', 'radius 0.01', 'radius 1']
    for i in performance:
        performance[i] = {j: None for j in estimation_methods}
        for j in performance[i]:
            performance[i][j] = {k: None for k in radii_indices}
            for k in performance[i][j]:
                if i == 'MV':
                    if k == 'standard':
                        result = portfolios['MV'][j].weights
                        print(result.index.size)
                    else:
                        result = portfolios['robMV'][j][float(k[7:])].weights
                if i == 'utility':
                    if k == 'standard':
                        result = portfolios['utility'][j].weights
                    else:
                        result = portfolios['robutility'][j][float(k[7:])].weights
                if i == 'CVaR':
                    if k == 'standard':
                        result = portfolios['CVaR'][j].weights
                    else:
                        result = portfolios['robCVaR'][j][float(k[7:])].weights
                performance[i][j][k] = (yearly_perf(result, yearly_returns).var(), total_perf(portfolio=result, returns=yearly_returns))

    save_pickle(performance, os.path.join(paths.PORTFOLIOS, experiment_name+'_performance'))
