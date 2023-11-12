import os
import datetime
import random

import pandas as pd
from matplotlib import pyplot as plt

import paths
from utility_functions.utils import load_market, save_pickle
from utility_functions.df_manipulation import timestep_consistency


class Market:

    """
    contains data and description of the market
    contains methods to visualize and access the information
    """

    def __init__(self, data_file_name: str, prices_or_returns: str, market_name: str, website=None):
        """
        :param data_file_name:
        :param prices_or_returns: 'prices' if we give price data, 'returns' if return data
        :param market_name: give a nome to the market
        :param website: a URL giving information about the data
        """
        self.data_file = data_file_name  # where data is stored
        self.name = market_name
        self.last_modified = datetime.datetime.now()
        self.website = website
        self.description = None
        self.comments = {'quality': None, 'format': None}
        self.prices = None
        self.returns = None

        while prices_or_returns not in ['prices', 'returns']:
            prices_or_returns = input(' Specify if the data are prices or returns: ')
        if prices_or_returns == 'prices':
            self.prices = pd.read_pickle(os.path.join(paths.CLEAN_DATA, self.data_file))
        else:
            self.returns = pd.read_pickle(os.path.join(paths.CLEAN_DATA, self.data_file))

    def monthly_returns(self):
        return self.prices.pct_change().iloc[1:]

    def monthly_returns_scaled_to_year(self):
        """ monthly returns multiplied by 365/days in month """
        """ careful: more data but more sensitive to monthly noise => larger variance """
        monthly_returns = self.prices.pct_change()
        for i, t in enumerate(monthly_returns.index[1:]):
            previous_t = monthly_returns.index[i]
            nb_days = t - previous_t
            nb_days = nb_days.days
            multiplier = 365 / nb_days
            monthly_returns.loc[t] = monthly_returns.loc[t] * multiplier
        yearly_returns = monthly_returns.iloc[1:]
        return yearly_returns

    def rolling_annual_returns(self):
        """ for each month look at yearly return of the previous year, careful: non i.i.d. data"""
        prices = self.prices
        returns = prices.copy(deep=True)
        if timestep_consistency(prices) is False:
            print('WARNING: timestep inconsistency')
        else:
            for i, t in enumerate(prices.index[11:]):
                for c in prices.columns:
                    p = prices[c][t]
                    p_previous = prices[c][i]
                    if p_previous is not None:
                        returns[c][t] = (p-p_previous)/p_previous
                        if c == 'S&P 500' and (p-p_previous)/p_previous > 0.3:
                            print(f' t-1: {prices.index[i]}     t: {t}')
            # remove the 11 first rows
            returns = returns.drop(returns.index[:11])
        return returns

    def calendar_year_returns(self, start_year=None):
        prices = self.prices
        # returns = prices.copy(deep=True)
        times = []
        indices = {}
        for i, t in enumerate(prices.index[11:]): # remove the 11 first
            if t.month == 12:
                times.append(t)
                indices[t] = i
        returns = pd.DataFrame(index=times, columns=prices.columns)
        for i, t in enumerate(returns.index):
            p_t = prices.loc[t]
            p_previous = prices.iloc[indices[t]-12]
            returns.loc[t] = (p_t - p_previous)/p_previous # TODO: normalize for 365 days ?
        if start_year is None:
            return returns
        else:
            return returns[returns.index >= pd.to_datetime(f'{start_year}-01-01')]

    def sp_annual_by_chatgpt(self):
        monthly_returns = self.prices['S&P 500']

        # Step 1: Aggregate monthly total returns to annual total returns
        annual_returns = monthly_returns.groupby(monthly_returns.index.year).prod()

        # Step 2: Convert annual total returns to percentage
        annual_returns_percent = (annual_returns - 1) * 100

        # Step 3: Compute annualized total return
        num_years = len(annual_returns)
        annualized_return = ((1 + annual_returns.prod()) ** (1 / num_years)) - 1

        annualized_return.plot.bar()
        plt.grid(True)
        date_labels = annualized_return.index.strftime('%Y-%m-%d')  # Format the timestamps as desired
        plt.xticks(range(len(annual_returns.index)), date_labels, rotation=45)
        plt.show()

    def assets(self):
        if self.prices is not None:
            return self.prices.columns
        elif self.returns is not None:
            return self.returns.columns

    def info(self):
        pass

    def plot(self):
        half_size = round(self.prices.columns.size/2)
        list_styles = ['-' for _ in range(half_size)]
        list_styles.extend(['--' for _ in range(self.prices.columns.size-half_size)])
        random.shuffle(list_styles)
        self.prices.plot(style=list_styles)
        plt.show()

    def keep_columns(self, list_of_columns_names: list):
        if len(list_of_columns_names) > 0:
            pass

    def date_increment(self):
        # return date increment like month
        pass

    def save(self, save_name=None, folder_path=paths.MARKETS):
        if save_name is not None:
            self.name = save_name
        self.last_modified = datetime.datetime.now()
        save_pickle(what=self, where=os.path.join(folder_path, self.name))


if __name__ == "__main__":
    m = Market(data_file_name='PublicData', prices_or_returns='prices', market_name='some_name')
    m.save()

    mm = load_market('some_name')

