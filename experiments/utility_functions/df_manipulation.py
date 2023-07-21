import pickle
import os
import shutil
from collections import Counter
import random

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from experiments import paths
from experiments.utility_functions.utils import load_market
import riskfolio.src.ParamsEstimation as pe
import riskfolio.src.AuxFunctions as af


def timestep_consistency(df):
    """ checks if the time between the time steps is below 35 days """
    consistent = True
    for i, t in enumerate(df.index[1:]):
        previous_t = df.index[i]
        nb_day = t - previous_t
        nb_day = nb_day.days
        if nb_day > 35:
            print(f'Warning: {previous_t}    {t}')
            consistent = False
    return consistent


def nan_values_distribution(df):
    """ for assets with nan values, give 0 if real value and >1 if nan """
    print(df.isna().sum())

    #
    df = df.applymap(lambda x: 0 if x < 99999 else x)
    # Reorder the DataFrame columns
    df = df[df.isna().sum().sort_values().index]
    df = df.fillna(1)
    for c in df.columns:
        if max(df[c]) < 1:
            df = df.drop(c, axis=1)

    for i, c in enumerate(df.columns):
        print(f' c: {c}   coef: {1+0.1*i}')
        df[c] = df[c] * (1 + 0.1*i)

    possible_styles = ['-', '-.', '--']
    list_styles = random.choices(possible_styles, k=df.columns.size)
    print(list_styles)
    df.plot(style=list_styles)


def calendar_year_returns(prices, start_year=None):
    # returns = prices.copy(deep=True)
    times = []
    indices = {}
    for i, t in enumerate(prices.index[11:]): # remove the 11 first
        if t.month == 12:
            times.append(t)
            indices[t] = i
    returns = pd.DataFrame(index=times, columns=prices.columns)
    for i, t in enumerate(returns.index):
        p_t = prices.loc[t]
        p_previous = prices.iloc[indices[t]-12]
        returns.loc[t] = (p_t - p_previous)/p_previous # TODO: normalize for 365 days ?
    if start_year is None:
        return returns
    else:
        return returns[returns.index >= pd.to_datetime(f'{start_year}-01-01')]


def get_mu_sigma(returns):
    returns = returns.apply(pd.to_numeric, errors='coerce')
    mu = pe.mean_vector(returns)
    cov = pe.covar_matrix(returns)
    value = af.is_pos_def(cov, threshold=1e-8)
    for i in range(5):
        if not value:
            try:
                cov = af.cov_fix(cov, method="clipped", threshold=1e-5)
                value = af.is_pos_def(cov, threshold=1e-8)
            except:
                break
        else:
            break

    if not value:
        pass
        # print("You must convert self.cov to a positive definite matrix")
    return mu, cov


def mu_cov_prev_years(returns, t, years):
    """ compute mu and cov from 'years' previous years"""
    cut_returns = returns[returns.index < t]
    if returns.index[1] - returns.index[0] > pd.Timedelta(days=32):
        # we have yearly returns
        cut_returns = cut_returns[-int(years):]
    else:
        cut_returns = cut_returns[-12*int(years):]
    return get_mu_sigma(cut_returns)


def mu_mean_reversion(returns, t, years):
    """ computes mu using previous years and the mean reversion assumption"""
    mu_past, _ = mu_cov_prev_years(returns, t, years)
    # use all previous data
    mu_total, cov_total = get_mu_sigma(returns[returns.index < t])
    return 2*mu_total - mu_past, cov_total


def yearly_perf(portfolio, returns):
    """

    Parameters
    ----------
    portfolio: pd.Dataframe
    returns: pd.Dataframe

    Returns
    -------

    """
    # create empty dataframe
    portfolio_returns = pd.Series(index=portfolio.index)
    for t in portfolio.index:
        portfolio_returns.loc[t] = sum(portfolio.loc[t] * returns.loc[t])
    return portfolio_returns


def total_perf(portfolio, returns):
    """ returns the return in percentage """
    portfolio_returns = yearly_perf(portfolio, returns)
    total = 1
    for r in portfolio_returns:
        total *= (1 + r)
    return (total-1)*100


if __name__ == '__main__':
    m = load_market('some_name')
    # plt.plot(m.prices['Gold'])
    # plt.show()
    nan_values_distribution(m.prices)
    plt.show()

