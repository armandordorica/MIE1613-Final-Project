import numpy as np
import pandas as pd
import timestamps as ts

import matplotlib.pyplot as plt



def common_random_numbers(size, seed=10):
    np.random.seed(seed)
    return np.random.rand(size)

import numpy as np

def get_montecarlo_paths(returns, close_df, close_var_name, num_paths=1000, forecast_days=30, seed=10):
    initial_price = close_df[close_var_name].iloc[-1]
    mean_daily_return = returns.mean()
    std_daily_return = returns.std()

    results = []

    # Generate common random numbers
    common_random_nums = common_random_numbers(num_paths * forecast_days, seed)
    common_random_nums = common_random_nums.reshape(num_paths, forecast_days)

    for i in range(num_paths):
        # Transform the common random numbers to normally distributed random numbers
        daily_returns = mean_daily_return + std_daily_return * np.random.normal(0, 1, size=forecast_days) * common_random_nums[i]
        prices = [initial_price]

        for daily_return in daily_returns:
            prices.append(prices[-1] * (1 + daily_return))

        results.append(prices)

    results = np.array(results)

    return results


# def get_montecarlo_paths(returns, close_df, close_var_name,   num_paths=1000, forecast_days=252,  title='', xlabel='',ylabel=''):

#     initial_price = close_df[close_var_name].iloc[-1]
#     mean_daily_return = returns.mean()
#     std_daily_return = returns.std()

#     results = []

#     for i in range(num_paths):
#         daily_returns = np.random.normal(mean_daily_return, std_daily_return, forecast_days)
#         prices = [initial_price]

#         for daily_return in daily_returns:
#             prices.append(prices[-1] * (1 + daily_return))

#         results.append(prices)

#     results = np.array(results)
    
#     return results


def plot_montecarlo_paths(returns, close_df, close_var_name,  num_paths=1000, forecast_days=30, title='', xlabel='',ylabel=''):
    initial_price = close_df[close_var_name].iloc[-1]
    
    # montecarlo_paths = get_montecarlo_paths(returns,  close_df, close_var_name,  num_paths, forecast_days,  title='', xlabel='',ylabel='')
    montecarlo_paths =get_montecarlo_paths(returns, close_df, close_var_name, num_paths=1000, forecast_days=forecast_days, seed=10)

    plt.figure(figsize=(20,10))
    plt.plot(pd.to_datetime(close_df.index),  close_df[close_var_name])

    start_date = pd.to_datetime(close_df.index)[-1]
    n = len(montecarlo_paths.T) # Generate 10 trading days
    trading_days = ts.get_trading_days(start_date, n)

    plt.plot(trading_days, montecarlo_paths.T)

    plt.xticks(rotation=90)
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    plt.title(title, fontsize=20)
    return montecarlo_paths


def plot_possible_mc_path_extension(input_df, single_sample_path, title='', xlabel='', ylabel=''): 
    plt.figure(figsize=(20,10))
    plt.plot(pd.to_datetime(input_df.index),  input_df['Close'])

    start_date = pd.to_datetime(input_df.index)[-1]
    n = len(single_sample_path) # Generate 10 trading days
    trading_days = ts.get_trading_days(start_date, n)

    plt.plot(trading_days, single_sample_path)
    plt.title(title, fontsize=20)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    