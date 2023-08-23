import os
from dro_analysis import paths
from dro_analysis.utility_functions.df_manipulation import calendar_year_returns
from dro_analysis.utility_functions.utils import save_pickle, load_pickle

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
import riskfolio as rp


def plot_differences(yearly_returns, test_times, radii_of_interest, portfolios, min_ret):
    # plot
    light_blue = mcolors.to_rgba('lightblue', alpha=1.0)
    medium_blue = mcolors.to_rgba('royalblue', alpha=1.0)
    dark_blue = mcolors.to_rgba('darkblue', alpha=0.8)
    darker_blue = mcolors.to_rgba('navy', alpha=1.0)
    col = [light_blue, medium_blue, dark_blue, darker_blue, 'black']
    bar_width = 0.25

    for radius_of_interest in radii_of_interest:
        fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(12, 9))
        for i, t in enumerate(test_times):
            years = 10
            cut_returns = yearly_returns[yearly_returns.index < t]
            cut_returns = cut_returns[-int(years):]
            exp_return = cut_returns.mean(axis=0)
            first_asset_above = exp_return[exp_return >= min_ret].sort_values().index[-1]
            if exp_return[exp_return <= min_ret].size >0:
                last_asset_below = exp_return[exp_return <= min_ret].sort_values().index[-1]
            else:
                last_asset_below =first_asset_above
            std_dev = cut_returns.std(axis=0)
            sorted_assets = std_dev.sort_values().index

            # MV
            wmv = portfolios['MV']['prev_10_years'].weights.loc[t]
            data = wmv.to_frame()
            count = 0
            for radius, port in portfolios['robMV']['prev_10_years'].items():
                if radius in radius_of_interest:
                    wr = port.weights.loc[t]
                    diff = wmv - wr
                    diff = diff.reindex(sorted_assets)
                    # data[str(radius)] = diff

                    index = range(len(diff))
                    axes[0, i].bar([j + count * bar_width for j in index], diff, bar_width, color=col[count],
                                   label=f'radius {radius}')
                    count += 1
            axes[0, i].axvline(x=last_asset_below, color='red', linestyle='--')
            axes[0, i].axvline(x=first_asset_above, color='red', linestyle='--')

            # Set the x-axis labels
            labels = ['MM', 'Tr', 'IG', 'HY', 'S&P', 'Nas', 'IDE', 'EM', 'HF', 'Com', 'Go']
            axes[0, i].set_xticks([j + bar_width for j in index])
            axes[0, i].set_xticklabels(labels)
            axes[1, i].set_xticks([j + bar_width for j in index])
            axes[1, i].set_xticklabels(labels)

            # utility
            wut = portfolios['utility']['prev_10_years'].weights.loc[t]
            data = wut.to_frame()
            count = 0
            for radius, port in portfolios['robutility']['prev_10_years'].items():
                if radius in radius_of_interest:
                    wr = port.weights.loc[t]
                    diff = wut - wr
                    diff = diff.reindex(sorted_assets)
                    # data[str(radius)] = diff

                    index = range(len(diff))
                    axes[1, i].bar([j + count * bar_width for j in index], diff, bar_width, color=col[count],
                                   label=f'radius {radius}')
                    count += 1

            axes[0, i].set_title(f'{t.year}')

        axes[0, 2].legend()
        axes[0, 0].set_ylabel('weights MV - weights robust MV')
        axes[1, 0].set_ylabel('weights utility - weights robust utility')
        axes[1, 1].set_xlabel('Assets in increasing order of variance over last 10 years')
        plt.show()


def plot_weights(yearly_returns, test_times, radii_of_interest, portfolios, min_ret):
    # plot
    light_blue = mcolors.to_rgba('lightblue', alpha=1.0)
    medium_blue = mcolors.to_rgba('royalblue', alpha=1.0)
    dark_blue = mcolors.to_rgba('darkblue', alpha=0.8)
    darker_blue = mcolors.to_rgba('navy', alpha=1.0)
    col = [light_blue, medium_blue, dark_blue, darker_blue, 'black']
    bar_width = 0.2

    for radius_of_interest in radii_of_interest:
        fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(12, 9))
        for i, t in enumerate(test_times):
            years = 10
            cut_returns = yearly_returns[yearly_returns.index < t]
            cut_returns = cut_returns[-int(years):]
            exp_return = cut_returns.mean(axis=0)
            first_asset_above = exp_return[exp_return >= min_ret].sort_values().index[-1]
            if exp_return[exp_return <= min_ret].size > 0:
                last_asset_below = exp_return[exp_return <= min_ret].sort_values().index[-1]
            else:
                last_asset_below = first_asset_above
            std_dev = cut_returns.std(axis=0)
            sorted_assets = std_dev.sort_values().index

            # MV
            wmv = portfolios['MV']['prev_10_years'].weights.loc[t]
            data = wmv.to_frame()
            count = 0
            for radius, port in portfolios['robMV']['prev_10_years'].items():
                if radius in radius_of_interest:
                    wr = port.weights.loc[t]
                    wr = wr.reindex(sorted_assets)
                    # data[str(radius)] = diff

                    index = range(len(wr))
                    axes[0, i].bar([j + count * bar_width for j in index], wr, bar_width, color=col[count],
                                   label=f'radius {radius}')
                    count += 1
            axes[0, i].bar([j + count * bar_width for j in index], wmv, bar_width, color='orange',
                           label=f'MV')
            axes[0, i].axvline(x=last_asset_below, color='red', linestyle='--')
            axes[0, i].axvline(x=first_asset_above, color='red', linestyle='--')

            # Set the x-axis labels
            labels = ['MM', 'Tr', 'IG', 'HY', 'S&P', 'Nas', 'IDE', 'EM', 'HF', 'Com', 'Go']
            axes[0, i].set_xticks([j + bar_width for j in index])
            axes[0, i].set_xticklabels(labels)
            axes[1, i].set_xticks([j + bar_width for j in index])
            axes[1, i].set_xticklabels(labels)

            # utility
            wut = portfolios['utility']['prev_10_years'].weights.loc[t]
            data = wut.to_frame()
            count = 0
            for radius, port in portfolios['robutility']['prev_10_years'].items():
                if radius in radius_of_interest:
                    wr = port.weights.loc[t]
                    wr = wr.reindex(sorted_assets)
                    # data[str(radius)] = diff

                    index = range(len(wr))
                    axes[1, i].bar([j + count * bar_width for j in index], wr, bar_width, color=col[count],
                                   label=f'radius {radius}')
                    count += 1
            axes[1, i].bar([j + count * bar_width for j in index], wut, bar_width, color='orange',
                           label=f'utility lambda=2')
            axes[0, i].set_title(f'{t.year}')

        axes[0, 2].legend()
        axes[0, 0].set_ylabel('weights MV')
        axes[1, 0].set_ylabel('weights utility')
        axes[1, 1].set_xlabel('Assets in increasing order of variance over last 10 years')
        plt.show()


if __name__ == "__main__":
    exp_to_analyze = 5
    if exp_to_analyze in [1, 3]:
        _min_ret = 0.05
    elif exp_to_analyze in [2, 4]:
        _min_ret = 0.1
    elif exp_to_analyze == 5:
        _min_ret = 0.06

    # get prices
    data_file = 'PublicData'
    prices = pd.read_pickle(os.path.join(paths.CLEAN_DATA, data_file))
    prices = prices.loc[prices.index > pd.Timestamp(1990, 1, 30)]  # no need to cut, already done in functions
    prices = prices.drop('TIPS US', axis=1)

    # get yearly returns
    _yearly_returns = prices.resample('Y').last().pct_change()

    _test_times = _yearly_returns.index[_yearly_returns.index.year.isin([2001, 2011, 2021])]
    _radii_of_interest = [[0.001, 0.01, 0.1]]

    _portfolios = load_pickle(os.path.join(paths.EXPERIMENTS, 'single_period_port_compo', f'exp_{exp_to_analyze}_portfolios'))
    _performance = load_pickle(os.path.join(paths.EXPERIMENTS, 'single_period_port_compo', f'exp_{exp_to_analyze}_performance'))

    plot_weights(_yearly_returns, _test_times, _radii_of_interest, _portfolios, _min_ret)
    # plot_differences(_yearly_returns, _test_times, _radii_of_interest, _portfolios, _min_ret)
