import os
from dro_analysis import paths
import random

import numpy as np
import pandas as pd
import riskfolio as rp
from dro_analysis.utility_functions.utils import save_pickle, load_pickle, get_returns
from dro_analysis.paths import PERFORMANCE_EVALUATION
import dro_analysis.experiments.performance_evaluation.config as config


def compute_w(df_returns, cov, target, obj, rm, kelly=False, radius=0):
    p = rp.Portfolio(returns=df_returns)
    # constraints
    if target is not None:
        if obj == 'MaxRet':
            target /= np.sqrt(52)
            if rm == 'MV':
                p.upperdev = target
            elif rm == 'robvariance':
                p.upperrobvariance = target
            elif rm == 'CVaR':
                p.upperCVaR = target
            elif rm == 'robCVaR':
                p.upperrobCVaR = target  # TODO:  !!!!! target has to be modified differently !!!!!
        elif obj == 'MinRisk':
            target /= 52
            if kelly == False:
                p.lowerret = target
            elif kelly == 'robmean':
                p.lowerrobret = target
    p.mu = df_returns.mean()
    p.cov = cov
    p.optimization(model='Classic', rm=rm, obj=obj, kelly=kelly, hist=True, radius=radius)

    try:
        return p.optimal['weights']
    except:
        return None


def compute_w_dro(df_returns, cov, target_risk, radius):
    p = rp.Portfolio(returns=df_returns, upperrobvariance=target_risk)
    p.mu = df_returns.mean()
    p.cov = cov
    p.optimization(model='Classic', rm='robvariance', obj='MaxRet', kelly='robmean', hist=True, radius=radius)

    return p.optimal['weights']


def compute_w_dro_l(df_returns, cov, target_risk, radius):
    """
    the 'l' stands for light, we maximize robust return with a constraint on empricial risk not robust risk
    """
    p = rp.Portfolio(returns=df_returns, upperdev=target_risk)
    p.mu = df_returns.mean()
    p.cov = cov
    p.optimization(model='Classic', rm='MV', obj='MaxRet', kelly='robmean', hist=True, radius=radius)

    return p.optimal['weights']


def generate_rolling(df, test_size, train_size, cov, target, radii, obj, risk_measure):
    """

    Parameters
    ----------
    df: weekly returns
    method:
    sampling_size: int, nuumber of samples to generate
    test_size: int, years
    train_size: int, years

    Returns
    -------

    """
    results = []
    last = df.index[-1] - pd.DateOffset(weeks=(test_size + train_size) * 52)

    times = []
    t = df.index[0]
    while t in df.index:
        times.append(t)
        t += pd.DateOffset(weeks=train_size * 52)

    for i, t in enumerate(times):
        print(f' --- {i + 1} / {len(times)}')
        df_train = df[t <= df.index]  # TODO: check time spread
        df_train = df_train[df_train.index < t + pd.DateOffset(weeks=train_size * 52)]
        df_test = df[t + pd.DateOffset(weeks=train_size * 52) <= df.index]
        df_test = df_test[df_test.index < t + pd.DateOffset(weeks=(test_size + train_size) * 52)]

        # # define portfolio class
        # w_mv = compute_w_mv(df_train, cov, target)
        # w_dro = {}
        # w_dro_l = {}
        # for radius in radii:
        #     w_dro[radius] = compute_w_dro(df_train, cov, target, radius)
        #     w_dro_l[radius] = compute_w_dro_l(df_train, cov, target, radius)
        # mu_train = df_train.mean()
        # mu_test = df_test.mean()
        # results.append({'w_mv': w_mv, 'w_dro': w_dro, 'w_dro_l': w_dro_l, 'mu_train': mu_train, 'mu_test': mu_test,
        #                 'train_start': t})

        # compute non dro portfolios
        if risk_measure == 'std':
            w = compute_w(df_train, cov, target, obj=obj, rm='MV')
        elif risk_measure == 'CVaR':
            w = compute_w(df_train, cov, target, obj=obj, rm='CVaR')

        # compute dro portfolios
        w_dro = {}
        w_dro_l = {}
        for radius in radii:
            if risk_measure == 'std':
                w_dro[radius] = compute_w(df_train, cov, target, obj=obj, rm='robvariance', kelly='robmean',
                                          radius=radius)
                w_dro_l[radius] = compute_w(df_train, cov, target, obj=obj, rm='MV', kelly='robmean', radius=radius)
            if risk_measure == 'CVaR':
                w_dro[radius] = compute_w(df_train, cov, target, obj=obj, rm='robCVaR', kelly='robmean', radius=radius)
                w_dro_l[radius] = compute_w(df_train, cov, target, obj=obj, rm='robCVaR', kelly=False, radius=radius)
        mu_train = df_train.mean()
        mu_test = df_test.mean()
        results.append(
            {'w_mv': w, 'w_dro': w_dro, 'w_dro_l': w_dro_l, 'mu_train': mu_train, 'mu_test': mu_test, 'train_start': t})

    return results


def generate_all_periods(df, test_size, train_size, cov, target, radii: list, obj, risk_measure):
    """

    Parameters
    ----------
    df
    test_size
    train_size
    cov:
    target_risk
    radii: list of radii

    Returns
    -------

    """
    results = []
    last = df.index[-1] - pd.DateOffset(weeks= (test_size+train_size) * 52)
    times = df[df.index <= last].index

    for i, t in enumerate(times):
        print(f' --- {i+1} / {times.size}')

        # make train/test
        df_train = df[t <= df.index] # TODO: check time spread
        df_train = df_train[df_train.index < t + pd.DateOffset(weeks= train_size * 52)]
        df_test = df[t + pd.DateOffset(weeks=train_size * 52) <= df.index ]
        df_test = df_test[df_test.index < t + pd.DateOffset(weeks= (test_size+train_size) * 52)]

        # # define portfolio class
        # w_mv = compute_w_mv(df_train, cov, target)
        # w_dro = {}
        # w_dro_l = {}
        # for radius in radii:
        #     w_dro[radius] = compute_w_dro(df_train, cov, target, radius)
        #     w_dro_l[radius] = compute_w_dro_l(df_train, cov, target, radius)
        # results.append({'w_mv': w_mv, 'w_dro': w_dro, 'w_dro_l': w_dro_l, 'mu_train': mu_train, 'mu_test': mu_test, 'train_start': t})

        # compute non dro portfolios
        if risk_measure == 'std':
            w = compute_w(df_train, cov, target, obj=obj, rm='MV')
        elif risk_measure == 'CVaR':
            w = compute_w(df_train, cov, target, obj=obj, rm='CVaR')

        # compute dro portfolios
        w_dro = {}
        w_dro_l = {}
        for radius in radii:
            if risk_measure == 'std':
                w_dro[radius] = compute_w(df_train, cov, target, obj=obj, rm='robvariance', kelly='robmean',
                                          radius=radius)
                w_dro_l[radius] = compute_w(df_train, cov, target, obj=obj, rm='MV', kelly='robmean', radius=radius)
            if risk_measure == 'CVaR':
                w_dro[radius] = compute_w(df_train, cov, target, obj=obj, rm='robCVaR', kelly='robmean', radius=radius)
                w_dro_l[radius] = compute_w(df_train, cov, target, obj=obj, rm='robCVaR', kelly=False, radius=radius)
        mu_train = df_train.mean()
        mu_test = df_test.mean()
        results.append({'w_mv': w, 'w_dro': w_dro, 'w_dro_l': w_dro_l, 'mu_train': mu_train, 'mu_test': mu_test, 'train_start': t})

    return results


def generate_boostrap(df, sample_size, train_window, test_window, cov, target, radii: list, obj, risk_measure):
    """
    for each draw:
        - randomly select a time
        - randomly select nb of years to train/test around the time, in the window between 1y and the value given above
    Parameters
    ----------
    df
    sample_size
    train_window:
    test_window
    cov
    target_risk
    radii

    Returns
    -------

    """
    results = []
    skipped = 0
    for i in range(sample_size):
        print(f' --- {i + 1} / {sample_size}')

        # get train/test
        train_size = random.randint(1, train_window)
        test_size = random.randint(1, test_window)
        last = df.index[-1] - pd.DateOffset(weeks=(test_size + train_size) * 52)
        times = df[df.index <= last].index
        t = random.choice(times)
        df_train = df[t <= df.index] # TODO: check time spread
        df_train = df_train[df_train.index < t + pd.DateOffset(weeks=train_size * 52)]
        df_test = df[t + pd.DateOffset(weeks=train_size * 52) <= df.index ]
        df_test = df_test[df_test.index < t + pd.DateOffset(weeks=(test_size+train_size) * 52)]

        # compute non dro portfolios
        if risk_measure == 'std':
            w = compute_w(df_train, cov, target, obj=obj, rm='MV')
        elif risk_measure == 'CVaR':
            w = compute_w(df_train, cov, target, obj=obj, rm='CVaR')

        # compute dro portfolios
        w_dro = {}
        w_dro_l = {}
        for radius in radii:
            if risk_measure == 'std':
                w_dro[radius] = compute_w(df_train, cov, target, obj=obj, rm='robvariance', kelly='robmean', radius=radius)
                if obj == 'MaxRet':
                    w_dro_l[radius] = compute_w(df_train, cov, target, obj=obj, rm='MV', kelly='robmean', radius=radius)
                elif obj == 'MinRisk':
                    w_dro_l[radius] = compute_w(df_train, cov, target, obj=obj, rm='robvariance', kelly=False, radius=radius)
            if risk_measure == 'CVaR':
                w_dro[radius] = compute_w(df_train, cov, target, obj=obj, rm='robCVaR', kelly='robmean', radius=radius)
                if obj == 'MaxRet':
                    w_dro_l[radius] = compute_w(df_train, cov, target, obj=obj, rm='CVaR', kelly='robmean', radius=radius)
                elif obj == 'MinRisk':
                    w_dro_l[radius] = compute_w(df_train, cov, target, obj=obj, rm='robCVaR', kelly=False,
                                                radius=radius)
        mu_train = df_train.mean()
        mu_test = df_test.mean()
        results.append({'w_mv': w, 'w_dro': w_dro, 'w_dro_l': w_dro_l, 'mu_train': mu_train, 'mu_test': mu_test, 'train_start': t})
    return results


def generate_all_periods_test(df, test_size, target_risk, radii, obj, rm):
    results = []
    last = df.index[-1] - pd.DateOffset(weeks=test_size * 52)
    times = df[df.index <= last].index

    w_mv = compute_w_mv(df, df.cov(), target_risk)
    w_dro = {}
    w_dro_l = {}
    for radius in radii:
        w_dro[radius] = compute_w_dro(df, df.cov(), target_risk, radius)
        w_dro_l[radius] = compute_w_dro_l(df, df.cov(), target_risk, radius)
    mu_train = df.mean()

    for i, t in enumerate(times):
        print(f' --- {i+1} / {times.size}')
        df_test = df[t <= df.index]  # TODO: check time spread
        df_test = df_test[df_test.index < t + pd.DateOffset(weeks=test_size * 52)]

        # define portfolio class
        mu_test = df_test.mean()
        results.append({'w_mv': w_mv, 'w_dro': w_dro, 'w_dro_l': w_dro_l, 'mu_train': mu_train, 'mu_test': mu_test, 'train_start': t})

    return results


def generate(years_30, radii, output_name, method, sample_size, test_years, train_years, target_yearly=None, obj='MaxRet', risk_measure='std'):
    """

    Parameters
    ----------
    sample_size
    years_30: bool, True if start at 1990, False for 1973
    radii: list of radii
    output_name: str, 'diplay_' + an int
    method: 'bootstrap', 'rolling', 'all_periods'
    sample_size int or None
    test_years: int or None
    train_years: int or None

    """
    target = target_yearly
    df = get_returns(years_30=years_30)     # TODO:  add target adj to all
    if method == 'bootstrap':
        res = generate_boostrap(df, sample_size=sample_size, train_window=train_years, test_window=test_years,
                                cov=df.cov(), target=target, radii=radii, obj=obj, risk_measure=risk_measure)
    elif method == 'rolling':
        res = generate_rolling(df, test_size=test_years, train_size=train_years, cov=df.cov(), target=target,
                               radii=radii, obj=obj, risk_measure=risk_measure)
    elif method == 'all_periods':
        res = generate_all_periods(df, test_size=test_years, train_size=train_years, cov=df.cov(),
                                   target=target,
                                   radii=radii, obj=obj, risk_measure=risk_measure)
    elif method == 'all_periods_test':
        res = generate_all_periods_test(df, test_size=test_years, target_risk=target, radii=radii, obj=obj, risk_measure=risk_measure)

    save_pickle(res, os.path.join(PERFORMANCE_EVALUATION, 'results', output_name))


if __name__ == "__main__":
    for exp in [config.exp_min_risk_10]:
        generate(**exp)
