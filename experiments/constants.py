TYPES = ['fixed return', 'fixed risk', 'only risk']
ALGOS = [{'markowitz'}]
RISK_MEASURES = ['Variance', 'VaR', 'CVaR', 'mean_cov', 'mean_std']

TRAIN_CHOICE = {'sliding': 'smth'}
CONDITIONING = {'none': None, 'quatratic': {}}   # maybe use a comp instead?


# THINK about conditioning on how the x last days looked like, maybe better performance

MARKET_PLOT_PARAMETERS = {'some_name': {'Money market US': {'color': 'blue', 'lw': 2, 'label': 'Money market US', 'mean': None, 'std_dev': None},
                                        'TIPS US': {'color': 'darkorange', 'lw': 1.5, 'label': 'TIPS US', 'mean': None, 'std_dev': None},
                                        'Treasuries US': {'color': 'gold', 'lw': 1.5, 'label': 'Treasuries US', 'mean': None, 'std_dev': None},
                                        'Investment Grade US': {'color': 'red', 'lw': 1.5, 'label': 'Invest. Grade US', 'mean': None, 'std_dev': None},
                                        'High yield US': {'color': 'black', 'lw': 1.5, 'label': 'High yield US', 'mean': None, 'std_dev': None},
                                        'S&P 500': {'color': 'black', 'lw': 1.5, 'label': 'S&P 500', 'mean': None, 'std_dev': None},
                                        'Nasdaq composite': {'color': 'black', 'lw': 1.5, 'label': 'Nasdaq composite', 'mean': None, 'std_dev': None},
                                        'International developed equities': {'color': 'black', 'lw': 1.5, 'label': 'Int. dev. equities', 'mean': None, 'std_dev': None},
                                        'Emerging markets': {'color': 'black', 'lw': 1.5, 'label': 'Emerging markets', 'mean': None, 'std_dev': None},
                                        'Hedge funds': {'color': 'black', 'lw': 1.5, 'label': 'Hedge funds', 'mean': None, 'std_dev': None},
                                        'Commodities': {'color': 'black', 'lw': 1.5, 'label': 'Commodities', 'mean': None, 'std_dev': None},
                                        'Gold': {'color': 'black', 'lw': 1.5, 'label': 'Gold', 'mean': None, 'std_dev': None}},
                          'new_market': {'td': 'td'}
                          }
