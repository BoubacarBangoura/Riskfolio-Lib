from abc import abstractmethod

import cvxpy as cp
import pandas as pd
import numpy as np

import financial_objects.portfolio as portfolio
from utility_functions.utils import load_market


class Risk:
    def __init__(self):
        self.description = None

    def cvxpy_format(self, past_returns, w: cp.Variable):
        """ generate line for the program """
        cvxpy_line = None
        return cvxpy_line

    @abstractmethod
    def compute(self, past_returns: pd.DataFrame, p: portfolio.Portfolio):
        pass


class Variance(Risk):
    def __init__(self):
        self.description = 'w^T*Sigma*w'

    def cvxpy_format(self, past_returns: pd.DataFrame, w: cp.Variable):
        return w.T @ past_returns.cov() @ w

    @staticmethod
    def compute(past_returns: pd.DataFrame, portfolio_weights: pd.Series):# p: financial_objects.portfolio.Portfolio):
        w = portfolio_weights.values
        cov = past_returns.cov().values
        np.dot(w.T, np.dot(cov, w))
        return np.dot(w.T, np.dot(cov, w))


class VaR(Risk):
    """ value at risk"""
    pass


class CVaR(Risk):
    """ conditional value at risk """
    pass


class MeanCov(Risk):
    pass


if __name__ == '__main__':
    m = load_market('some_name')
    r = m.returns[['S&P 500', 'Commodities']]
    _w = pd.Series([0.5, 0.5])
    print(r.cov())
    print(r.cov().values)
    print(Variance.compute(r, _w))
