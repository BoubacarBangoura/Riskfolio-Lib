import os
import random
import matplotlib.pyplot as plt
import numpy as np

from dro_analysis.utility_functions.utils import load_pickle, get_returns
from dro_analysis.paths import PERFORMANCE_EVALUATION
import dro_analysis.experiments.performance_evaluation.config as config
from dro_analysis.experiments.performance_evaluation.analyze import get_distr_ret


def compounded_yearly_investement(invest_years, farming_years, return_rate=0.05, amount_per_year=None, rand=False, std=None):
    """
    continuous investment
    Returns
    -------

    """
    val = 0
    for i in range(invest_years):
        if rand:
            f_r = random.normalvariate(return_rate, std)
        else:
            f_r = return_rate
        val = (val + amount_per_year) * (1 + f_r)

    for i in range(farming_years):
        if rand:
            f_r = random.normalvariate(return_rate, std)
        else:
            f_r = return_rate
        val *= (1 + f_r)
    return val


def compounded_yearly_investement_dro(invest_years, farming_years, distribution, amount_per_year=None):
    """
    continuous investment
    Returns
    -------

    """
    val = 0
    for i in range(invest_years):
        f_r = random.choice(distribution)
        val = (val + amount_per_year) * (1 + f_r)

    for i in range(farming_years):
        f_r = random.choice(distribution)
        val *= (1 + f_r)
    return val


def plot_gaussian(return_rate=0.098, inv_years=[5, 10, 10], farm_years=[5, 5, 10], std=0.16, amount_per_year=50):

    to_plot = {}
    fig, axs = plt.subplots(1, len(inv_years))
    for i, y in enumerate(farm_years):
        to_plot[y] = []
        for _ in range(10000):
            to_plot[y].append(compounded_yearly_investement(inv_years[i], y, return_rate=return_rate,
                                                           amount_per_year=amount_per_year, rand=True, std=std))
        print('-------------------------------------------------')
        axs[i].hist(to_plot[y], bins=50, alpha=0.7)
        axs[i].axvline(x=np.mean(to_plot[y]), color='black', linestyle='--', label='mean')
        axs[i].axvline(x=np.quantile(to_plot[y], 0.25), color='orange', linestyle='--', label='quantile 0.25')
        axs[i].axvline(x=np.quantile(to_plot[y], 0.75), color='orange', linestyle='--')
        axs[i].axvline(x=np.quantile(to_plot[y], 0.05), color='red', linestyle='--', label='quantile 0.05')
        axs[i].axvline(x=np.quantile(to_plot[y], 0.95), color='red', linestyle='--')
        axs[i].axvline(x=np.quantile(to_plot[y], 0.02), color='magenta', linestyle='--', label='quantile 0.01')
        axs[i].axvline(x=np.quantile(to_plot[y], 0.98), color='magenta', linestyle='--')
        axs[i].set_title(f'invest: {inv_years[i]} y  farm: {farm_years[i]} y  inv val: {inv_years[i]*amount_per_year} k')
        axs[i].grid()
        axs[i].legend()

        y_ticks = axs[i].get_yticks()
        axs[i].set_yticklabels(['{:.0f}%'.format(100 * yy / len(to_plot[y])) for yy in y_ticks])
    plt.suptitle(f'')
    plt.show()


def plot_dro(distribution, inv_years=[5, 10, 10], farm_years=[5, 5, 10], amount_per_year=50):
    to_plot = {}
    fig, axs = plt.subplots(1, len(inv_years))
    for i, y in enumerate(farm_years):
        to_plot[y] = []
        for _ in range(10000):
            to_plot[y].append(compounded_yearly_investement_dro(inv_years[i], y, distribution, amount_per_year=amount_per_year))
        axs[i].hist(to_plot[y], bins=50, alpha=0.7)
        axs[i].axvline(x=np.mean(to_plot[y]), color='black', linestyle='--', label='mean')
        axs[i].axvline(x=np.quantile(to_plot[y], 0.25), color='orange', linestyle='--', label='quantile 0.25')
        axs[i].axvline(x=np.quantile(to_plot[y], 0.75), color='orange', linestyle='--')
        axs[i].axvline(x=np.quantile(to_plot[y], 0.05), color='red', linestyle='--', label='quantile 0.05')
        axs[i].axvline(x=np.quantile(to_plot[y], 0.95), color='red', linestyle='--')
        axs[i].axvline(x=np.quantile(to_plot[y], 0.02), color='magenta', linestyle='--', label='quantile 0.01')
        axs[i].axvline(x=np.quantile(to_plot[y], 0.98), color='magenta', linestyle='--')
        axs[i].set_title(f'invest: {inv_years[i]} y  farm: {farm_years[i]} y  inv val: {inv_years[i]*amount_per_year} k')
        axs[i].grid()
        axs[i].legend()

        y_ticks = axs[i].get_yticks()
        axs[i].set_yticklabels(['{:.0f}%'.format(100 * yy / len(to_plot[y])) for yy in y_ticks])
    plt.suptitle(f'')
    plt.show()


if __name__ == "__main__":
    choice = 3
    if choice == 1:
        pass
    elif choice == 2:
        plot_gaussian(return_rate=0.098, inv_years=[5, 10, 10], farm_years=[10, 5, 10], std=0.1434, amount_per_year=50)
    elif choice == 3:
        conf = config.exp_15
        df = get_returns(conf['years_30'])
        res = load_pickle(os.path.join(PERFORMANCE_EVALUATION, 'results', conf['output_name']))
        d = {'strategy': 'dro',   'sample': 'test',  'radius': 5e-6, 'robust_pred': False}
        distribution_rets = get_distr_ret(res, **d)
        distribution_rets = [val*52 for val in distribution_rets]
        plot_dro(distribution=distribution_rets, inv_years=[5, 10, 10], farm_years=[10, 5, 10], amount_per_year=50)
        plot_dro(distribution=distribution_rets, inv_years=[1, 1, 1], farm_years=[5, 10, 15], amount_per_year=500)
    elif choice == 4:
        conf = config.exp_15
        df = get_returns(conf['years_30'])
        res = load_pickle(os.path.join(PERFORMANCE_EVALUATION, 'results', conf['output_name']))
        d = {'strategy': 'mv',    'sample': 'test',  'radius': 5e-6, 'robust_pred': False}
        distribution_rets = get_distr_ret(res, **d)
        distribution_rets = [val * 52 for val in distribution_rets]
        plot_dro(distribution=distribution_rets, inv_years=[1, 1, 1], farm_years=[5, 10, 15], amount_per_year=500)




