from abc import abstractmethod

import pandas as pd
from dro_analysis import constants


class Strategy:
    def __init__(self, opt_parameters={}, port_parameters={}, estimation_method={}, name=None):
        self.opt_parameters = opt_parameters
        self.port_parameters = port_parameters
        self.estimation_method = estimation_method
        self.name = name
        self.complete_name = self.make_name()

    def make_name(self):
        if self.name is None:
            return None
        else:
            name = self.name + str(self.opt_parameters.get('l', '')) + str(self.opt_parameters.get('radius', ''))
            return name



class Markowitz(Strategy):
    def __init__(self, risk_metric, past_data_use, conditioning: dict):
        pass

    @abstractmethod
    def optimize(self, expected_returns: pd.Series, covariance: pd.DataFrame):
        pass


class MarkowitzFixedReturn(Markowitz):
    def __init__(self):
        pass

    def optimize(self, expected_returns: pd.Series, covariance: pd.DataFrame):
        pass


class MeanCovFixedReturn(MarkowitzFixedReturn):
    def __init__(self):
        self.risk_measure = 'variance'

    def optimize(self, expected_returns: pd.Series, covariance: pd.DataFrame):
        w = None


        return w


class MarkowitzFixedRisk(Markowitz):
    def __init__(self):
        pass


class MarkowitzRiskOnly(Markowitz):
    def __init__(self):
        pass


class BL(Strategy):
    def __init__(self):
        pass


class DROFixedRisk(Strategy):
    def __init__(self):
        pass


class MarkowitzRiskOnly(Strategy):
    def __init__(self):
        pass


def markowitz(exp_return, cov_matrix, risk: str, opt_type): # fixed_return_risk_or_nothing):
    # choose which of the program below to run
    pass


def mean_cov_fixed_return(exp_return, cov_matrix):
    pass


def mean_cov_fixed_risk(exp_return, cov_matrix):
    # calls the right program with respect to risk
    pass


def mean_cov_fixed_var(exp_return, cov_matrix):
    pass


def black_litterman(something):
    pass


def dro(something_as_well, **kwargs):
    pass


def equal_weighting(stuff):
    pass
