from utility_functions.utils import load_portfolio


class AnalyzePortfolios:
    def __init__(self, portfolios_names: list):
        self.portfolios = []
        self.add_portfolios(portfolios_names)

    def add_portfolios(self, portfolios_names: list):
        if len(portfolios_names) > 0:
            for name in portfolios_names:
                p = load_portfolio(name)
                self.portfolios.append(p)


if __name__ == '__main__':
    a = AnalyzePortfolios(['la'])
    a.add_portfolios(['lala', 'lalala'])
    for _p in a.portfolios:
        print(_p.name)

