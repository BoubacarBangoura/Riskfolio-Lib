import os
import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import dro_analysis.paths as paths


def available_data(raw_prices):
    df = raw_prices

    # Sort columns by the number of available data points
    sorted_columns = df.count().sort_values(ascending=True).index
    df = df[sorted_columns]

    # Create a color map for lines
    colors = plt.cm.viridis(np.linspace(0, 1, len(df.columns)))

    # Initialize a figure and axis
    fig, ax = plt.subplots()

    # Iterate through columns and plot horizontal lines where data is available
    for i, (column, color) in enumerate(zip(df.columns, colors)):
        available_data = df[column].notna()
        y_level = i + 1
        ax.plot(df.index[available_data], np.full(sum(available_data), y_level), color=color, marker='o', label=column)

    # Set y-ticks and labels for each column
    ax.set_yticks(range(1, len(df.columns) + 1))
    ax.set_yticklabels(df.columns)

    # Set the legend
    # ax.legend()

    # Set labels and title
    ax.set_xlabel('Time')
    ax.set_title('Available Data for different asset classes')

    # Show the plot
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def clean_prices(raw_prices):
    prices = raw_prices.loc[raw_prices.index > pd.Timestamp(1990, 1, 30)]  # no need to cut, already done in functions
    prices = prices.drop('TIPS US', axis=1)
    return prices


if __name__ == "__main__":
    # get prices
    data_file = 'PublicData2'
    prices = pd.read_pickle(os.path.join(paths.CLEAN_DATA, data_file))

    available_data(prices)
    prices = clean_prices(prices)

    # get yearly returns
    yearly_returns = prices.resample('Y').last().pct_change()



