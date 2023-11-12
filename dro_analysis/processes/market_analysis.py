import matplotlib.pyplot as plt

import paths
from utility_functions.utils import load_market
from utility_functions.df_manipulation import timestep_consistency, nan_values_distribution
from constants import MARKET_PLOT_PARAMETERS
from financial_objects.market import Market


class MarketAnalysis:

    def __init__(self, market_name, save_dir=paths.MARKETS):
        self.market = load_market(market_name)

    def missing_data_info(self, returns: str):
        """  """
        df = getattr(self.market, f'{returns}')()
        timestep_consistency(df)
        nan_values_distribution(df)

    def error_bar(self, returns: str, start_year=None):
        """

        :param start_year:
        :param returns: 'monthly_returns', 'yearly_returns_by_month', ...
        :return:
        """
        df = getattr(self.market, f'{returns}')(start_year)
        # df = df.dropna()
        fig, axs = plt.subplots(1, figsize=(10, 10))
        to_plot = MARKET_PLOT_PARAMETERS[self.market.name]

        # # compare distributions
        # for name in to_plot:
        #     data = df[name]
        #     x, y = kde(data)
        #     to_plot[name]['mean'] = data.mean()
        #     to_plot[name]['std_dev'] = data.std()
        #     axs[0, 0].plot(x, y, color=to_plot[name]['color'], lw=to_plot[name]['lw'],
        #                    label=f'{to_plot[name]["label"]}')
        #     # axs[i, 0].axvline(mean, color=to_plot[name], linestyle='-', linewidth=1.5, alpha=0.7)
        #
        # axs[0, 0].legend()
        # axs[0, 0].grid()
        # plt.xticks(rotation=20)

        # mean and standard deviation
        for name in to_plot:
            data = df[name]
            to_plot[name]['mean'] = data.mean()
            to_plot[name]['std_dev'] = data.std()

        sorted_assets = sorted(to_plot.items(), key=lambda x: x[1]['mean'])
        for asset in sorted_assets:
            name = asset[0]
            print(f'{name} std dev: {to_plot[name]["std_dev"]}')
            axs.errorbar(to_plot[name]['label'], to_plot[name]['mean'], yerr=to_plot[name]['std_dev'], fmt='o',
                               capsize=4, c=to_plot[name]['color'])
        axs.set_title('Increasing mean')
        axs.grid()
        axs.set_xticklabels(axs.get_xticklabels(), rotation=10)

        fig.suptitle(f'Distribution of the assets')
        plt.show()

    def box_plot(self, returns: str):
        df = getattr(self.market, f'{returns}')()
        # df = df.dropna()
        fig, axs = plt.subplots(1, figsize=(10, 10))
        to_plot = MARKET_PLOT_PARAMETERS[self.market.name]

        # mean and standard deviation
        for name in to_plot:
            data = df[name]
            to_plot[name]['mean'] = data.mean()
            to_plot[name]['std_dev'] = data.std()

        # boxplot
        labels = []
        for name in to_plot:
            labels.append(to_plot[name]['label'])
        data = [df[name].tolist() for name in to_plot]
        axs.boxplot(data, labels=labels)
        axs.set_title('Boxplots of the assets')
        axs.grid()
        plt.xticks(rotation=20)
        plt.show()


if __name__ == "__main__":
    m = MarketAnalysis('some_name')

    # ana.plot_forecast_prices_crash()
    # ana.plot_prod()
    # ana.plot_spot()
    # for _to_plot in ['Up Regulation', 'Down Regulation', 'Spot price']:
    #     ana.fit_distribution(_to_plot)
    # ana.fit_distribution('Spot price')
    # ana.prices_distribution()
    # ana.productions_distribution()


    # r = m.market.calendar_year_returns(start_year=2000)['S&P 500']
    # r.plot.bar()
    # plt.grid(True)
    # date_labels = r.index.strftime('%Y-%m-%d')  # Format the timestamps as desired
    # plt.xticks(range(len(r.index)), date_labels, rotation=45)

    m.market.sp_annual_by_chatgpt()


    # m.missing_data_info(returns='rolling_yearly_returns')
    # m.error_bar(returns='calendar_year_returns', start_year=1996)
    # m.box_plot(returns='calendar_year_returns')

    plt.show()
