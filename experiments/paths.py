"""
Paths to folders containing code, data and other resources like images, plots and text
"""
import os

HOME_PATH = os.path.expanduser('~')
# replace PROJECT by your own project directory
PROJECT = os.path.join(HOME_PATH, 'PycharmProjects', 'Riskfolio-Lib', 'experiments')
DATA = os.path.join(PROJECT, 'data')
RAW_DATA = os.path.join(DATA, 'raw_data')
CLEAN_DATA = os.path.join(DATA, 'clean_data')
MARKETS = os.path.join(DATA, 'markets')
PORTFOLIOS = os.path.join(DATA, 'portfolios')


# CLUSTER_DATA = os.path.join(HOME_PATH, 'DRO_optimal_nomination', 'Data')
# CLUSTER_RESULTS = os.path.join(CLUSTER_DATA, 'Results')
