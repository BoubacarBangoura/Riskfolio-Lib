import pickle
import os
import shutil
from collections import Counter
import random

from experiments import paths

import numpy as np
import scipy.stats as st
import matplotlib.pyplot as plt


def make_discrete_dist(observations):
    """
    takes a list or pd.Series and makes a discrete distribution out of it in a dictionary format
    :param observations: list or pd.Series
    :return: dict
    """
    freq_dict = Counter(observations)
    total = len(observations)
    return {k: v / total for k, v in freq_dict.items()}


def gaussian(data, nb_points=1000):
    """ fit a gaussian and return x and y values for the plot """
    mu, std = st.norm.fit(data)
    spread = max(data) - min(data)
    start = round(min(data) - 0.1 * spread) - 1
    end = round(max(data) + 0.1 * spread)
    x = np.linspace(start, end, nb_points)
    p = st.norm.pdf(x, mu, std)
    return x, p


def kde(data, nb_points=1000):
    spread = max(data) - min(data)
    start = min(data) - 0.1 * spread
    end = max(data) + 0.1 * spread
    x = np.linspace(start, end, nb_points)
    return x, st.gaussian_kde(data).evaluate(x)


def pareto(data, nb_points=1000):
    alpha, loc, scale = st.pareto.fit(data)
    pareto_dist = st.pareto(alpha, loc=loc, scale=scale)
    spread = max(data) - min(data)
    start = round(min(data) - 0.1 * spread) - 1
    end = round(max(data) + 0.1 * spread)
    x = np.linspace(start, end, nb_points)
    return x, pareto_dist.pdf(x)
