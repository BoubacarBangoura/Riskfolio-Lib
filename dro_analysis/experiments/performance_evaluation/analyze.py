import os
import random

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm, colors
import numpy as np
import seaborn as sns

from dro_analysis import paths
from dro_analysis.utility_functions.utils import save_pickle, load_pickle, get_returns
from dro_analysis.paths import PERFORMANCE_EVALUATION
import dro_analysis.experiments.performance_evaluation.config as config
from dro_analysis.utility_functions.distributions import make_discrete_dist, kde, pareto, gaussian

color = {0: 'blue', 1: 'red', 2: 'darkorange', 3: 'orange', 4: 'yellow', 5: 'pink', 6: 'green', 7:  'magenta', 8: 'black'}


def get_distr_ret(results_data, strategy, sample, radius, robust_pred):
    """

    Parameters
    ----------
    results: str
    strategy: 'mv', 'dro', 'dro_l'
    sample: 'test' or 'train'
    radius: float
    robust: True for robust return, False otherwise

    sample = 'train' + radius for robust prediction

    Returns
    -------

    """
    rets = []
    for r in results_data:
        if strategy == 'mv':
            w = r['w_mv']
        elif strategy == 'dro':
            w = r['w_dro'][radius]
        elif strategy == 'dro_l':
            w = r['w_dro_l'][radius]
        if w is not None:
            val = np.dot(w, r[f'mu_{sample}'])
            if sample == 'train' and robust_pred:
                val -= np.sqrt(radius) * np.linalg.norm(w)
            if strategy == 'mv' and sample == 'train':
                print(f'val: {val*52}')
            rets.append(val)

    return rets


def get_distr_std(results_data, strategy, sample, radius, train_size, test_size, years_30):
    """

    Parameters
    ----------
    results: str
    strategy: 'mv', 'dro', 'dro_l'
    sample: 'test' or 'train'
    radius: float
    robust: True for robust return, False otherwise

    sample = 'train' + radius for robust prediction

    Returns
    -------

    """
    std = []
    count = 0
    for r in results_data:
        count += 1
        print(count)
        if strategy == 'mv':
            w = r['w_mv']
        elif strategy == 'dro':
            w = r['w_dro'][radius]
        elif strategy == 'dro_l':
            w = r['w_dro_l'][radius]
        if w is not None:
            # ----------------- new block -------------
            # get df test returns
            df = get_returns(years_30=years_30)
            t = r['train_start']
            df_test = df[t + pd.DateOffset(weeks=train_size * 52) <= df.index]
            df_test = df_test[df_test.index < t + pd.DateOffset(weeks=(test_size + train_size) * 52)]
            test_returns = np.dot(w, df_test.T)

            # take std
            std.append(np.std(test_returns))
            # -----------------------------------------

    return std


def get_distr_risk(results_data, strategy, radius, cov: pd.DataFrame):
    """

        Parameters
        ----------
        results_data: str
        strategy: 'mv', 'dro', 'dro_l'
        radius: float

        Returns
        -------

        """
    cov = cov.values
    risks = []
    for r in results_data:
        if strategy == 'mv':
            w = r[f'w_mv']
        elif strategy == 'dro':
            w = r['w_dro'][radius]
        elif strategy == 'dro_l':
            w = r['w_dro_l'][radius]
        if w is not None:
            val = np.sqrt(np.dot(w, np.dot(cov, w)))
            risks.append(val)

    return risks

def get_distr_weights(results_data, strategy, radius, exp):
    """
    get a df of the weights for all the runs of a given strategy

    Parameters
    ----------
    results: str
    strategy: 'mv', 'dro', 'dro_l'
    sample: 'test' or 'train'
    radius: float
    exp: experiment config
    """
    # get empty df
    all_weights = get_returns(years_30=exp['years_30'])
    all_weights.drop(all_weights.index, inplace=True)

    # fill up df
    for r in results_data:
        if strategy == 'mv':
            w = r['w_mv']
        elif strategy == 'dro':
            w = r['w_dro'][radius]
        elif strategy == 'dro_l':
            w = r['w_dro_l'][radius]
        all_weights.loc[len(all_weights)] = w

    return all_weights


def get_distr_reliability(results_data, strategy, radius, percentage=False):
    """
    """
    reli = []
    for r in results_data:
        if strategy == 'mv':
            w = r['w_mv']
            extra = 0
        else:  # dro, dro_l
            w = r[f'w_{strategy}'][radius]
            extra = - np.sqrt(radius) * np.linalg.norm(w)
        val = np.dot(w, r['mu_train']-r['mu_test']) + extra
        if percentage:
            val /= np.dot(w, r['mu_train'])
        reli.append(val)

    return reli


def returns_distribution(results, display, nb_bins=150):
    fig, axs = plt.subplots(2, 2)

    for d in display:
        color = config.get_color(d)
        label = config.get_label(d)
        rets = get_distr_ret(results, **d)
        rets = [val*52 for val in rets]
        if d['sample'] == 'train':
            row = 0
        else:
            row = 1

        axs[row, 0].hist(rets, bins=nb_bins, label=label, alpha=0.6, color=color, density=True)
        axs[row, 1].axvline(x=np.mean(rets), linestyle='--', linewidth=1.5, color=color)
        axs[row, 1].axvline(x=np.quantile(rets, 0.05), color='red', linestyle='--', label='quantile 0.05')
        axs[row, 1].axvline(x=np.quantile(rets, 0.95), color='red', linestyle='--')

        x, y = kde(rets)
        axs[row, 1].plot(x, y, color=color, label=label)

        x, y = gaussian(rets)
        axs[row, 1].plot(x, y, linestyle='--', color=color, linewidth=1.5)

    axs[0, 0].set_title('Average return ')
    axs[0, 1].set_title('In sample returns distribution: w*mu_in')
    axs[1, 0].set_title('Standard deviation of returns on test')
    axs[1, 1].set_title('Out-of-sample returns distribution: w*mu_out (-penalty)')

    axs[0, 1].set_ylabel('Frequency')
    axs[1, 1].set_ylabel('Frequency')

    axs[1, 1].set_xlabel('Annualized returns')
    for i in [0, 1]:
        for j in [0, 1]:
            axs[i, j].grid()
            axs[i, j].legend()
    fig.suptitle('')
    plt.show()


def plot_left_tail(results, display, alpha=0.05, nb_bins=50):
    fig, axs = plt.subplots(2, 2)
    rows = {'train': 0, 'test': 1}

    for d in display:
        color = config.get_color(d)
        label = config.get_label(d)
        rets = get_distr_ret(results, **d)
        rets = [val*52 for val in rets]
        rets.sort()
        rets = rets[0: int(np.floor(alpha*len(rets)))]

        row = rows[d['sample']]

        axs[row, 0].hist(rets, bins=nb_bins, label=label, alpha=0.4, density=True)
        axs[row, 1].axvline(x=np.mean(rets), linestyle='--', linewidth=1.5, color=color)

        x, y = kde(rets)
        axs[row, 1].plot(x, y, label=label)

        # x, y = gaussian(rets)
        #axs[row, 1].plot(x, y, linestyle='--', color=color, linewidth=1.5)

    axs[0, 0].set_title('Average return ')
    axs[0, 1].set_title('In sample returns distribution: w*mu_in')
    axs[1, 0].set_title('Standard deviation of returns on test')
    axs[1, 1].set_title('Out-of-sample returns distribution: w*mu_out (-penalty)')

    axs[0, 1].set_ylabel('Frequency')
    axs[1, 1].set_ylabel('Frequency')

    axs[1, 1].set_xlabel('Annualized returns')
    for i in [0, 1]:
        for j in [0, 1]:
            axs[i, j].grid()
            axs[i, j].legend()
    fig.suptitle('')
    plt.show()


def bar_plots(results, display, df, exp=None, tail_alpha=None):
    """

    Parameters
    ----------
    results
    display
    df
    exp: experiment
    tail_alpha:   eg 0.05, keep the tail only, applied on returns not the risks list

    Returns
    -------

    """
    fig, axs = plt.subplots(2, 2)

    stats = {'means': {},  'stds': {}, 'dist': {}}
    visited = {}
    for d in display:
        color = config.get_color(d)
        label = config.get_label(d)
        rets = get_distr_ret(results, **d)
        risks = get_distr_risk(results, d['strategy'], d['radius'], df.cov())
        rets = [val*52 for val in rets]
        if tail_alpha is not None:
            rets.sort()
            rets = rets[0: int(np.floor(tail_alpha * len(rets)))]

        # get 3 lists for the average returns
        if not f'{d["strategy"]}_{d["radius"]}' in stats['means']:       # if visit the strat for first time
            stats['means'][f'{d["strategy"]}_{d["radius"]}'] = [0, 0, 0]
        if d['sample'] == 'train' and d['robust_pred'] is False:         # strat case 1
            stats['means'][f'{d["strategy"]}_{d["radius"]}'][0] = np.mean(rets)
        elif d['sample'] == 'train' and d['robust_pred'] is True:        # strat case 2
            stats['means'][f'{d["strategy"]}_{d["radius"]}'][1] = np.mean(rets)
        elif d['sample'] == 'test':                                      # strat case 3
            stats['means'][f'{d["strategy"]}_{d["radius"]}'][2] = np.mean(rets)

        # get 3 lists for the stds
        if not f'{d["strategy"]}_{d["radius"]}' in stats['stds']:
            stats['stds'][f'{d["strategy"]}_{d["radius"]}'] = [0, 0, 0]
        if d['sample'] == 'train':
            if not f'{d["strategy"]}_{d["radius"]}_{d["sample"]}' in visited:
                stats['stds'][f'{d["strategy"]}_{d["radius"]}'][1] = np.std(rets)
                visited[f'{d["strategy"]}_{d["radius"]}_{d["sample"]}'] = 1
        elif d['sample'] == 'test':
            stats['stds'][f'{d["strategy"]}_{d["radius"]}'][0] = np.mean(risks) * np.sqrt(52)
            stats['stds'][f'{d["strategy"]}_{d["radius"]}'][2] = np.std(rets)

        # get 3 lists for the average returns
        if not f'{d["strategy"]}_{d["radius"]}' in stats['dist']:
            stats['dist'][f'{d["strategy"]}_{d["radius"]}'] = [[], [], []]
        if d['sample'] == 'train' and d['robust_pred'] is False:
            stats['dist'][f'{d["strategy"]}_{d["radius"]}'][0] = rets
        elif d['sample'] == 'train' and d['robust_pred'] is True:
            stats['dist'][f'{d["strategy"]}_{d["radius"]}'][1] = rets
        elif d['sample'] == 'test':
            stats['dist'][f'{d["strategy"]}_{d["radius"]}'][2] = rets

    bar_width = 0.25

    # bar mean

    ret_train = []
    ret_robtrain = []
    ret_test = []
    ret_labels = []

    for key, val in stats['means'].items():
        ret_train.append(val[0]*100)
        ret_robtrain.append(val[1]*100)
        ret_test.append(val[2]*100)
        ret_labels.append(key)

    x = np.arange(len(ret_labels))
    axs[0, 0].bar(x, ret_train, bar_width, label='train', color='lightblue')
    axs[0, 0].bar(x + bar_width, ret_robtrain, bar_width, label='train rob', color='steelblue')
    axs[0, 0].bar(x + 2*bar_width, ret_test, bar_width, label='test', color='blue')
    axs[0, 0].set_xticks(x + bar_width, ret_labels)
    axs[0, 0].legend()
    axs[0, 0].set_ylabel('%')

    if exp is not None:
        if exp['obj'] == 'MinRisk' and exp['target_yearly'] is not None:
            axs[0, 0].axhline(y=exp['target_yearly'] * 100, linestyle='-', label='Target', linewidth=1.5, color='darkblue')

    # bar std

    risk_sigma = []
    risk_std_train = []
    risk_std_test = []
    risk_labels = []

    for key, val in stats['stds'].items():
        risk_sigma.append(val[0]*100)
        risk_std_train.append(val[1]*100)
        risk_std_test.append(val[2]*100)
        risk_labels.append(key)

    x = np.arange(len(risk_labels))
    axs[1, 0].bar(x, risk_sigma, bar_width, label='sigma', color='pink')
    axs[1, 0].bar(x + bar_width, risk_std_train, bar_width, label='train', color='lightcoral')
    axs[1, 0].bar(x + 2 * bar_width, risk_std_test, bar_width, label='test', color='red')
    if exp is None:
        if exp['obj'] == 'MaxRet' and exp['target_yearly'] is not None:
            target_risk = exp['target_yearly'] * 100
            axs[1, 0].axhline(y=target_risk, linestyle='-', label='Target', linewidth=1.5, color='darkorange')

    axs[1, 0].set_xticks(x + bar_width, risk_labels)
    axs[1, 0].legend()
    axs[1, 0].set_ylabel('%')

    # boxes

    dist_train = []
    dist_robtrain = []
    dist_test = []
    dist_labels = []

    for key, val in stats['dist'].items():   # the *100 to have in percentage
        dist_train.append(val[0]*100)
        dist_robtrain.append(val[1]*100)
        dist_test.append(val[2]*100)
        dist_labels.append(key)

    box_lists = [dist_train, dist_robtrain, dist_test]
    positions = np.array(range(len(dist_test))) + 1
    box_width = 0.2
    colors_box = ['lightblue', 'steelblue', 'blue']

    for i, group_data in enumerate(box_lists):
        positions_shifted = positions + (i - 1) * box_width
        axs[0, 1].boxplot(group_data, positions=positions_shifted, widths=box_width, boxprops=dict(color=colors_box[i]))

    axs[0, 1].set_title('Boxplots')
    axs[0, 1].set_ylabel('Values')
    axs[0, 1].set_xticks(positions + bar_width, dist_labels)
    axs[0, 1].legend()

    axs[1, 1].boxplot(dist_test, positions=positions, widths=0.7, boxprops=dict(color='blue'))
    axs[1, 1].set_xticks(positions, dist_labels)

    # plots parameters

    axs[0, 0].set_title('Average return ')
    axs[0, 1].set_title('Boxplot: train-robtrain-test')
    axs[1, 0].set_title('Standard deviation of returns on test')
    axs[1, 1].set_title('Boxplot just test')

    axs[1, 1].set_xlabel('strategies')

    axs[0, 0].grid()
    axs[0, 1].grid()
    axs[1, 0].grid()
    axs[1, 1].grid()

    axs[0, 0].legend(loc='upper left')
    axs[1, 0].legend(loc='upper right')

    fig.suptitle('')
    plt.show()


def plot_weights_dist(results, display, exp=None):
    """

    Parameters
    ----------
    results
    display
    df
    exp: experiment
    tail_alpha:   eg 0.05, keep the tail only, applied on returns not the risks list

    Returns
    -------

    """
    # fig, axs = plt.subplots(1, len(display), sharey=True, sharex=True)
    fig, axs = plt.subplots(2, 2, sharey=True, sharex=True)
    coor = {0: [0, 0], 1: [0, 1], 2: [1, 0], 3: [1, 1]}

    for i, d in enumerate(display):
        ax = axs[coor[i][0], coor[i][1]]
        color = config.get_color(d)
        label = config.get_label(d)

        weights = get_distr_weights(results, d['strategy'], d['radius'], exp)

        sns.violinplot(data=weights, ax=ax, inner='point', cut=0)   # inner=“box”, “quart”, “point”, “stick”
        ax.set_ylabel('probability')
        title = f'{d["strategy"]}'
        if d["strategy"] == 'dro':
            title += f'   radius: {d["radius"]}'
        ax.set_title(title)
        ax.grid()
    fig.suptitle(f'Portfolio weights per strategy')
    plt.show()


def performance_per_draw(results, display):
    fig, axs = plt.subplots(1, 2, sharey=True)

    for d in display:
        color = config.get_color(d)
        label = config.get_label(d)
        rets = get_distr_ret(results, **d)
        rets = [val*52 for val in rets]
        axs[0].set_title('In sample performance per draw')
        axs[1].set_title('Out-of-sample performance per draw')
        if d['sample'] == 'train':
            row = 0
        else:
            row = 1
        axs[row].plot(rets, label=label, color=color)
    axs[0].set_ylabel('Annualized returns')
    axs[0].grid()
    axs[1].grid()
    axs[0].legend()
    axs[1].legend()
    fig.suptitle('')
    plt.show()


def reliability_plot(results, display, percentage=False, nb_bins=150):

    fig, axs = plt.subplots(1, 2)
    i = 0
    for d in display:
        reli = get_distr_reliability(results_data=results, strategy=d['strategy'], radius=d['radius'], percentage=percentage)
        label = config.get_label(display=d, plot='reli')
        axs[0].plot(reli, label=label)
        axs[1].hist(reli, bins=nb_bins, label=label, color=color[i])
        axs[1].axvline(x=np.mean(reli), color=color[i], linestyle='--', label='mean', linewidth=3)
        i += 1
    axs[0].grid()
    axs[1].grid()
    axs[0].set_title('value of w*(mu_in - mu_out) per draw')
    axs[1].set_title('distribution of w*(mu_in - mu_out)')
    plt.suptitle('w*(mu_in - mu_out)')
    plt.legend()
    plt.show()


def std_dev_distribution(results, display, target_risk, df, nb_bins=150):
    target_risk *= np.sqrt(52)
    size = len(display) + len(display) % 2
    nb_rows = int(size/2)
    fig, axs = plt.subplots(nb_rows, nb_rows)

    color = {0: 'blue', 1: 'red', 2: 'green', 3: 'yellow'}
    k = 0
    for i in range(nb_rows):
        for j in range(nb_rows):
            d = display[k]
            vals = get_distr_risk(results_data=results, strategy=d['strategy'], radius=d['radius'], cov=df.cov())
            vals = [val*np.sqrt(52) for val in vals]

            axs[i, j].hist(vals, bins=nb_bins, label=config.get_label(d, plot='risk'))
            axs[i, j].axvline(x=np.mean(vals), color=color[2], linestyle='--', label='mean', linewidth=2)
            axs[i, j].axvline(x=target_risk, color='orange', linestyle='--', label='target', linewidth=2)

            y_ticks = axs[i, j].get_yticks()
            axs[i, j].set_yticklabels(['{:.0f}%'.format(100*y/len(vals)) for y in y_ticks])
            axs[i, j].set_title(config.get_label(d, plot='risk'))
            axs[i, j].set_ylabel('Frequency')
            axs[i, j].set_xlabel('std: wT.Sigma.w')
            axs[i, j].grid()
            axs[i, j].legend()
            k += 1

    fig.suptitle('Distribution of wT.Sigma.w for Sigma the empirical over whole time')
    plt.legend()
    plt.show()


def ef(results, display, df, target_risk):
    color = {0: 'blue', 1: 'red', 2: 'green', 3: 'yellow'}
    target_risk *= np.sqrt(52)

    size = len(display) + len(display) % 2
    nb_rows = int(size/2)
    fig, axs = plt.subplots(nb_rows, nb_rows)

    k = 0
    for i in range(nb_rows):
        for j in range(nb_rows):
            d = display[k]
            stds = get_distr_risk(results_data=results, strategy=d['strategy'], radius=d['radius'], cov=df.cov())
            stds = [val*np.sqrt(52) for val in stds]
            rets = get_distr_ret(results_data=results, strategy=d['strategy'], sample=d['sample'], radius=d['radius'], robust_pred=d['robust_pred'])
            rets = [val * 52 for val in rets]
            label = config.get_label(d)

            axs[i, j].scatter(stds, rets, label=label, s=1)
            axs[i, j].axvline(x=np.mean(stds), color='green', linestyle='--', label='mean stds', linewidth=2)
            axs[i, j].axhline(y=np.mean(rets), color='blue', linestyle='--', label='mean rets', linewidth=2)
            axs[i, j].axvline(x=target_risk, color='yellow', linestyle='--', label='target', linewidth=2)

            axs[i, j].set_title(label)
            axs[i, j].grid()
            axs[i, j].legend()
            k += 1
        axs[1, i].set_xlabel('std')
        axs[i, 0].set_ylabel('exp return')
    fig.suptitle('EFs')
    plt.legend()
    plt.show()


def make_list_from_res():
    """
    take the results and make a dictionary with list of mean, std and label for each experiement (train/test/method)
    Returns
    -------

    """
    for dro in ['dro', 'dro_l']:
        for tgV in [0.075, 0.05, 0.1]:
            for method in ['all_periods']:  # 'rolling',
                # one plot
                # fig, axs = plt.subplots(2, 4, sharex=True, sharey=True)

                k = 0
                for train_y in [1, 2, 3, 5, 10]:
                    if train_y >= 5:
                        list_test = [1, 2]
                    else:
                        list_test = [1]
                    for test_y in list_test:
                        exp = {'years_30': False,
                               'radii': [5e-7, 1e-6, 5e-6, 1e-5, 5e-5],
                               'output_name': f'tgV_{tgV}_{train_y}+{test_y}_{method}',
                               'method': method,
                               'sample_size': 100,
                               'test_years': test_y,
                               'train_years': train_y,
                               'target_yearly': tgV,
                               'obj': 'MaxRet',
                               'risk_measure': 'std'}

                        results = load_pickle(
                            os.path.join(PERFORMANCE_EVALUATION, 'delta_consistency', exp['output_name']))

                        list_means, list_stds, list_labels, list_nb_data = [], [], [], []

                        rets = get_distr_ret(results, strategy='mv', sample='test', radius=5e-5, robust_pred=False)
                        rets = [val * 52 for val in rets]
                        rets = [val for val in rets if not np.isnan(val)]  # remove nans

                        stds = get_distr_std(results, strategy='mv', sample='test', radius=5e-5, train_size=train_y,
                                             test_size=test_y, years_30=False)
                        stds = [val * 52**0.5 for val in stds]
                        stds = [val for val in stds if not np.isnan(val)]  # remove nans

                        list_nb_data.append(len(rets))
                        list_means.append(np.mean(rets) * 100)
                        list_stds.append(np.mean(stds) * 100)
                        list_labels.append('mv')
                        for rad in exp['radii']:
                            rets = get_distr_ret(results, strategy=dro, sample='test', radius=rad, robust_pred=False)
                            rets = [val * 52 for val in rets]
                            rets = [val for val in rets if not np.isnan(val)]  # remove nans

                            stds = get_distr_std(results, strategy=dro, sample='test', radius=rad, train_size=train_y,
                                                 test_size=test_y, years_30=False)
                            stds = [val * 52**0.5 for val in stds]
                            stds = [val for val in stds if not np.isnan(val)]  # remove nans

                            list_nb_data.append(len(rets))
                            list_means.append(np.mean(rets)*100)
                            list_stds.append(np.mean(stds)*100)   # TODO: update
                            list_labels.append(f'{rad}')

                        dict_res = {'means': list_means, 'stds': list_stds, 'labels': list_labels, 'nb_data': list_nb_data}
                        save_pickle(dict_res, os.path.join(PERFORMANCE_EVALUATION, 'delta_consistency',
                                                           f'DICT_{dro}_tgV_{tgV}_{train_y}+{test_y}_{method}'))


def check_wSw():
    """
    check that
    -------

    """
    df = get_returns(years_30=False)
    cov = df.cov()

    for tgV in [0.075, 0.05, 0.1]:
        target = tgV / np.sqrt(52)
        print(f' -----------------------------   tg {tgV}   --------------------------')
        for method in ['all_periods']:  # 'rolling',
            # one plot
            # fig, axs = plt.subplots(2, 4, sharex=True, sharey=True)

            k = 0
            for train_y in [1, 2, 3, 5, 10]:
                # input('pause')
                if train_y >= 5:
                    list_test = [1, 2]
                else:
                    list_test = [1]
                for test_y in list_test:
                    exp = {'years_30': False,
                           'radii': [5e-7, 1e-6, 5e-6, 1e-5, 5e-5],
                           'output_name': f'tgV_{tgV}_{train_y}+{test_y}_{method}',
                           'method': method,
                           'sample_size': 100,
                           'test_years': test_y,
                           'train_years': train_y,
                           'target_yearly': tgV,
                           'obj': 'MaxRet',
                           'risk_measure': 'std'}

                    results = load_pickle(
                        os.path.join(PERFORMANCE_EVALUATION, 'delta_consistency', exp['output_name']))

                    count = 0
                    for r in results:
                        # count += 1
                        # print(count)

                        # w = r['w_mv']
                        # if isinstance(w, pd.Series):
                        #     val = np.dot(w, np.dot(cov, w)) ** 0.5
                        #     if val > target + 1e-6:
                        #         print(f' -- MV: {val * 52**0.5} -- ')

                        for radius in exp['radii']:
                            # w = r['w_dro'][radius]
                            # if isinstance(w, pd.Series):
                            #     val = np.dot(w, np.dot(cov, w))**0.5 + np.sqrt(radius) * np.linalg.norm(w, 2)
                            #     if val > target + 1e-6:
                            #         print(f' -- DRO: {val * 52**0.5} -- ')
                            #
                            w = r['w_dro_l'][radius]
                            if isinstance(w, pd.Series):
                                val = np.dot(w, np.dot(cov, w))**0.5
                                if val > target + 1e-6:
                                    print(f' -- DRO l: {val * 52**0.5} -- ')


def delta_choice_analysis():
    for method in ['all_periods', 'rolling']:
        for tgV in [0.05, 0.075, 0.1]:
            for dro in ['dro_l']:  # 'dro',
                # one plot
                fig, axs = plt.subplots(2, 4, sharex=True, sharey=True)

                k = 0
                for train_y in [1, 2, 3, 5, 10]:
                    if train_y >= 5:
                        list_test = [1, 2]
                    else:
                        list_test = [1]
                    for test_y in list_test:
                        exp = {'years_30': False,
                                       'radii': [5e-7, 1e-6, 5e-6, 1e-5, 5e-5],
                                       'output_name': f'tgV_{tgV}_{train_y}+{test_y}_{method}',
                                       'method': method,
                                       'sample_size': 100,
                                       'test_years': test_y,
                                       'train_years': train_y,
                                       'target_yearly': tgV,
                                       'obj': 'MaxRet',
                                       'risk_measure': 'std'}

                        dict_res = load_pickle(
                            os.path.join(PERFORMANCE_EVALUATION, 'delta_consistency', f'DICT_{dro}_tgV_{tgV}_{train_y}+{test_y}_{method}'))
                        list_means = dict_res['means']
                        list_stds = dict_res['stds']
                        list_labels = dict_res['labels']
                        list_nb_data = dict_res['nb_data']

                        bar_width = 0.3
                        if k < 4:
                            i = 0
                        else:
                            i = 1
                        j = k%4
                        x = np.arange(len(list_labels))
                        axs[i, j].bar(x, list_means, bar_width, label='mean', color='blue')
                        axs[i, j].bar(x + bar_width, list_stds, bar_width, label='std', color='red')
                        axs[i, j].axhline(y=tgV * 100, linestyle='--', label='Target', linewidth=1.5, color='darkorange')
                        axs[i, j].set_xticks(x + bar_width, list_labels)
                        axs[i, j].legend()
                        axs[i, j].set_ylabel('%')
                        title = f'Train: {train_y} y   Test: {test_y} y  '
                        if all(x == list_nb_data[0] for x in list_nb_data):
                            title += f'\nnb data : {list_nb_data[0]}'
                        else:
                            title += f'\nnb data :{list_nb_data[:3]} \n           {list_nb_data[3:]}'
                        axs[i, j].set_title(title)
                        axs[i, j].grid()

                        k += 1

                title = f'PROBLEM: maximize return        STD TARGET: {tgV*100} %        METHOD: '
                if method == 'rolling':
                    title += 'disjoint sets'
                else:
                    title += 'rolling window'
                title += '          ROBUSTNESS: '
                if dro == 'dro_l':
                    title += 'only return'
                else:
                    title += 'both risk and return'
                fig.text(0.5, 0.03, 'increasing radius', ha='center', va='center')
                fig.suptitle(title, fontweight='bold')
                fig.subplots_adjust(hspace=0.35)
                plt.show()

def make_list_from_res_extreme():
    """
    take the results and make a dictionary with list of mean, std and label for each experiement (train/test/method)
    Returns
    -------

    """
    for dro in ['dro', 'dro_l']:
        for tgV in [0.075, 0.1]:
            for method in ['all_periods']:
                # one plot
                # fig, axs = plt.subplots(2, 4, sharex=True, sharey=True)

                k = 0
                for train_y in [1, 2, 3, 5, 10]:
                    if train_y >= 5:
                        list_test = [1, 2]
                    else:
                        list_test = [1]
                    for test_y in list_test:
                        exp = {'years_30': False,
                               'radii': [5e-4, 1e-4, 1e-7, 1e-8],
                               'output_name': f'XTRM_tgV_{tgV}_{train_y}+{test_y}_{method}',
                               'method': method,
                               'sample_size': 100,
                               'test_years': test_y,
                               'train_years': train_y,
                               'target_yearly': tgV,
                               'obj': 'MaxRet',
                               'risk_measure': 'std'}

                        results = load_pickle(
                            os.path.join(PERFORMANCE_EVALUATION, 'delta_consistency', exp['output_name']))

                        list_means, list_stds, list_labels, list_nb_data = [], [], [], []

                        rets = get_distr_ret(results, strategy='mv', sample='test', radius=5e-5, robust_pred=False)
                        rets = [val * 52 for val in rets]
                        rets = [val for val in rets if not np.isnan(val)]  # remove nans

                        stds = get_distr_std(results, strategy='mv', sample='test', radius=5e-5, train_size=train_y, test_size=test_y)  # Todo: UPDATED

                        list_nb_data.append(len(rets))
                        list_means.append(np.mean(rets) * 100)
                        list_stds.append(np.mean(stds) * 100)  # TODO: updateD
                        list_labels.append('mv')
                        for rad in exp['radii']:
                            rets = get_distr_ret(results, strategy=dro, sample='test', radius=rad, robust_pred=False)
                            rets = [val * 52 for val in rets]
                            rets = [val for val in rets if not np.isnan(val)]  # remove nans
                            list_nb_data.append(len(rets))
                            list_means.append(np.mean(rets)*100)
                            list_stds.append(np.std(rets)*100)
                            list_labels.append(f'{rad}')

                        dict_res = {'means': list_means, 'stds': list_stds, 'labels': list_labels, 'nb_data': list_nb_data}
                        save_pickle(dict_res, os.path.join(PERFORMANCE_EVALUATION, 'delta_consistency',
                                                           f'DICT_XTRM_{dro}_tgV_{tgV}_{train_y}+{test_y}_{method}'))


def delta_choice_analysis_extreme():
    for method in ['all_periods']:
        for tgV in [0.075, 0.1]:
            for dro in ['dro', 'dro_l']:
                # one plot
                fig, axs = plt.subplots(2, 4, sharex=True, sharey=True)

                k = 0
                for train_y in [1, 2, 3, 5, 10]:
                    if train_y >= 5:
                        list_test = [1, 2]
                    else:
                        list_test = [1]
                    for test_y in list_test:
                        exp = {'years_30': False,
                                       'radii': [5e-4, 1e-4, 1e-7, 1e-8],
                                       'output_name': f'XTRM_tgV_{tgV}_{train_y}+{test_y}_{method}',
                                       'method': method,
                                       'sample_size': 100,
                                       'test_years': test_y,
                                       'train_years': train_y,
                                       'target_yearly': tgV,
                                       'obj': 'MaxRet',
                                       'risk_measure': 'std'}

                        dict_res = load_pickle(
                            os.path.join(PERFORMANCE_EVALUATION, 'delta_consistency', f'DICT_XTRM_{dro}_tgV_{tgV}_{train_y}+{test_y}_{method}'))
                        list_means = dict_res['means']
                        list_stds = dict_res['stds']
                        list_labels = dict_res['labels']
                        list_nb_data = dict_res['nb_data']

                        bar_width = 0.3
                        if k < 4:
                            i = 0
                        else:
                            i = 1
                        j = k%4
                        x = np.arange(len(list_labels))
                        axs[i, j].bar(x, list_means, bar_width, label='mean', color='blue')
                        axs[i, j].bar(x + bar_width, list_stds, bar_width, label='std', color='red')
                        axs[i, j].axhline(y=tgV * 100, linestyle='--', label='Target', linewidth=1.5, color='darkorange')
                        axs[i, j].set_xticks(x + bar_width, list_labels)
                        axs[i, j].legend()
                        axs[i, j].set_ylabel('%')
                        title = f'Train: {train_y} y   Test: {test_y} y  '
                        if all(x == list_nb_data[0] for x in list_nb_data):
                            title += f'\nnb data : {list_nb_data[0]}'
                        else:
                            title += f'\nnb data :{list_nb_data[:3]} \n           {list_nb_data[3:]}'
                        axs[i, j].set_title(title)
                        axs[i, j].grid()

                        k += 1

                title = f'PROBLEM: maximize return        STD TARGET: {tgV*100} %        METHOD: '
                if method == 'rolling':
                    title += 'disjoint sets'
                else:
                    title += 'rolling window'
                title += '          ROBUSTNESS: '
                if dro == 'dro_l':
                    title += 'only return'
                else:
                    title += 'both risk and return'
                fig.text(0.5, 0.03, 'increasing radius', ha='center', va='center')
                fig.suptitle(title, fontweight='bold')
                fig.subplots_adjust(hspace=0.35)
                plt.show()


if __name__ == "__main__":

    # conf = config.exp_TEST_cvar
    # df = get_returns(conf['years_30'])
    # res = load_pickle(os.path.join(PERFORMANCE_EVALUATION, 'results', conf['output_name']))
    # target_risk = df.std().mean()

    choice = 'min_risk_10'    # exp_1 - classic - cvar - cvar_2 - cvar_3 - max_ret - min_risk - min_risk_10 - exp_cvar_tg

    if choice == 'exp_2':
        conf = config.exp_2
        df = get_returns(conf['years_30'])
        res = load_pickle(os.path.join(PERFORMANCE_EVALUATION, 'results', conf['output_name']))

        print('target risk:', round(100*df.std().mean()*np.sqrt(52), 2), ' %')

        _display = config.display_15_bar
        bar_plots(res, _display, df, exp=conf)
        # bar_plots(res, _display, df, exp=conf, tail_alpha=0.05)
        # performance_per_draw(res, _display)

        _display = config.display_violin_TEST_2
        plot_weights_dist(res, _display, exp=conf)

        _display = config.display_15_rets
        returns_distribution(res, _display)
        plot_left_tail(res, _display, alpha=0.05)

        # _display = config.display_15_reli
        # reliability_plot(res, _display, percentage=False)
        # std_dev_distribution(res, _display, target_risk, df, nb_bins=150)
        # ef(res, _display, df, target_risk)
    elif choice == 'exp_1':
        conf = config.exp_1
        df = get_returns(conf['years_30'])
        res = load_pickle(os.path.join(PERFORMANCE_EVALUATION, 'results', conf['output_name']))
        print('target risk:', round(100*df.std().mean()*np.sqrt(52), 2), ' %')

        _display = config.display_2_bar
        bar_plots(res, _display, df)
    elif choice == 'cvar':
        conf = config.exp_TEST_cvar
        df = get_returns(conf['years_30'])
        res = load_pickle(os.path.join(PERFORMANCE_EVALUATION, 'results', conf['output_name']))

        _display = config.bar_cvar
        bar_plots(res, _display, df, exp=conf)
        bar_plots(res, _display, df, exp=conf, tail_alpha=0.05)

        _display = config.display_15_rets
        # returns_distribution(res, _display)
        plot_left_tail(res, _display, alpha=0.05)

        _display = config.display_violin
        plot_weights_dist(res, _display, exp=conf)
    elif choice == 'cvar_2':
        conf = config.exp_TEST_cvar_2
        df = get_returns(conf['years_30'])
        res = load_pickle(os.path.join(PERFORMANCE_EVALUATION, 'results', conf['output_name']))

        _display = config.bar_cvar_2
        bar_plots(res, _display, df, exp=conf)
        bar_plots(res, _display, df, exp=conf, tail_alpha=0.05)

        _display = config.display_violin_2
        plot_weights_dist(res, _display, exp=conf)
    elif choice == 'cvar_3':
        conf = config.exp_TEST_cvar_3
        df = get_returns(conf['years_30'])
        res = load_pickle(os.path.join(PERFORMANCE_EVALUATION, 'results', conf['output_name']))

        _display = config.bar_cvar_3
        bar_plots(res, _display, df, exp=conf)
        bar_plots(res, _display, df, exp=conf, tail_alpha=0.05)

        _display = config.display_violin_3
        plot_weights_dist(res, _display, exp=conf)
    elif choice == 'max_ret':
        conf = config.exp_max_ret
        df = get_returns(conf['years_30'])
        res = load_pickle(os.path.join(PERFORMANCE_EVALUATION, 'results', conf['output_name']))

        print('target risk:', round(100*df.std().mean()*np.sqrt(52), 2), ' %')

        # for _display in [config.bar_max_ret, config.bar_l_max_ret]:
        #     bar_plots(res, _display, df, exp=conf)
        #     bar_plots(res, _display, df, exp=conf, tail_alpha=0.05)
        #     # performance_per_draw(res, _display)

        _display = config.violin_max_ret
        plot_weights_dist(res, _display, exp=conf)

        # for _display in [config.rets_max_ret, config.rets_l_max_ret]:
        #     # returns_distribution(res, _display)
        #     plot_left_tail(res, _display, alpha=0.05)

        # _display = config.display_15_reli
        # reliability_plot(res, _display, percentage=False)
        # std_dev_distribution(res, _display, target_risk, df, nb_bins=150)
        # ef(res, _display, df, target_risk)
    elif choice == 'min_risk':
        conf = config.exp_min_risk
        df = get_returns(conf['years_30'])
        res = load_pickle(os.path.join(PERFORMANCE_EVALUATION, 'results', conf['output_name']))

        for _display in [config.bar_min_risk, config.bar_l_min_risk]:
            bar_plots(res, _display, df, exp=conf)
            bar_plots(res, _display, df, exp=conf, tail_alpha=0.05)
            # performance_per_draw(res, _display)

        # for _display in [config.rets_min_risk, config.rets_l_min_risk]:
        #     # returns_distribution(res, _display)
        #     plot_left_tail(res, _display, alpha=0.05)

        for _display in [config.violin_min_risk, config.violin_l_min_risk]:
            plot_weights_dist(res, _display, exp=conf)

        # _display = config.display_15_reli
        # reliability_plot(res, _display, percentage=False)
        # std_dev_distribution(res, _display, target_risk, df, nb_bins=150)
        # ef(res, _display, df, target_risk)
    elif choice == 'min_risk_10':
        conf = config.exp_min_risk_10
        df = get_returns(conf['years_30'])
        res = load_pickle(os.path.join(PERFORMANCE_EVALUATION, 'results', conf['output_name']))

        for _display in [config.bar_min_risk, config.bar_l_min_risk]:
            bar_plots(res, _display, df, exp=conf)
            bar_plots(res, _display, df, exp=conf, tail_alpha=0.05)
            # performance_per_draw(res, _display)

        # for _display in [config.rets_min_risk, config.rets_l_min_risk]:
        #     # returns_distribution(res, _display)
        #     plot_left_tail(res, _display, alpha=0.05)

        for _display in [config.violin_min_risk, config.violin_l_min_risk]:
            plot_weights_dist(res, _display, exp=conf)

        # _display = config.display_15_reli
        # reliability_plot(res, _display, percentage=False)
        # std_dev_distribution(res, _display, target_risk, df, nb_bins=150)
        # ef(res, _display, df, target_risk)
    elif choice == 'exp_cvar_tg':
        conf = config.exp_cvar_tg
        df = get_returns(conf['years_30'])
        res = load_pickle(os.path.join(PERFORMANCE_EVALUATION, 'results', conf['output_name']))

        for _display in [config.bar_cvar_tg, config.bar_l_cvar_tg]:
            bar_plots(res, _display, df, exp=conf)
            bar_plots(res, _display, df, exp=conf, tail_alpha=0.05)

        _display = config.display_violin_cvar_tg
        plot_weights_dist(res, _display, exp=conf)

    elif choice == 'delta_choice_analysis':
        # make_list_from_res()
        delta_choice_analysis()

        # check_wSw()

        # make_list_from_res_extreme()
        # delta_choice_analysis_extreme()
