import os
import datetime
import importlib

import pandas as pd

from experiments import constants
from experiments import paths
from experiments.financial_objects.Strategy import Strategy
from experiments.utility_functions.utils import save_pickle


class Portfolios:

    """ portfolio following a given strategy over a time window """

    def __init__(self, weights: pd.DataFrame, datafile: str, test_start: pd.Timestamp, strategy: Strategy, name=None):
        self.weights = weights # assets as index
        self.test_start = test_start
        self.datafile = datafile
        self.strategy = strategy
        self.name = name
        self.last_modified = datetime.datetime.now()

    def for_a_given_time_t_give_the_time_window_used(self):
        # use data_file and strategy
        pass

    def average_return(self):
        pass

    def total_return(self):
        pass

    def train_window(self):
        # does it make sense?
        pass

    def test_window(self):
        pass

    def risk(self, risk_measure: str): # does it make sense? Should I keep?
        # check is correct
        while risk_measure not in constants.RISK_MEASURES:
            risk_measure = input(f'select a risk measure among {constants.RISK_MEASURES} for portfolio {self.name}: ')

        risk_module = importlib.import_module("risk")
        risk_class = getattr(risk_module, risk_measure)

    def roi(self):
        # a serie with roi for each time of test window
        pass

    def info(self):
        pass

    def save(self, save_name=None, folder_path=paths.PORTFOLIOS):
        if save_name is not None:
            self.name = save_name
        self.last_modified = datetime.datetime.now()
        save_pickle(what=self, where=os.path.join(folder_path, self.name))


if __name__ == "__main__":
    _name = 'lalala'
    w = 1
    s = Strategy(train_choice='sliding', conditioning='none')
    p = Portfolios(weights=w, market_name='some_name', strategy=s, name=_name)
    p.save()

    p.risk('variance')
