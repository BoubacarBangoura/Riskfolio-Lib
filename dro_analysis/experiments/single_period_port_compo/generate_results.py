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

if __name__ == "__main__":
    experiment_name = 'testos'

    # get prices
    data_file = 'PublicData'
    prices = pd.read_pickle(os.path.join(paths.CLEAN_DATA, data_file))
    prices = prices.loc[prices.index > pd.Timestamp(1990, 1, 30)]  # no need to cut, already done in functions
    prices = prices.drop('TIPS US', axis=1)

    # get monthly and yearly returns
    monthly_returns = prices.pct_change().dropna()
    yearly_returns = calendar_year_returns(prices)
    yearly_returns = yearly_returns.drop(yearly_returns.index[0])

    strategies = {'MV': MV, 'robMV': robMV, 'utility': utility, 'robutility': robutility}
    estimation_methods = {'prev_10_years': prev_10_years}

    radii = [0.01, 1]
    _test_start = pd.to_datetime('2001-12-31')

