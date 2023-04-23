import numpy as np


def get_sharpe_ratio(returns, risk_free_rate=0.02):
    num_periods_per_year = 252  # Assuming 252 trading days in a year
    risk_free_rate_daily = np.power(1 + risk_free_rate, 1 / num_periods_per_year) - 1
    
    returns = np.asarray(returns)
    excess_returns = returns - risk_free_rate_daily
    return excess_returns.mean() / excess_returns.std()


def get_sortino_ratio(returns, risk_free_rate=0.02, target_rate=0.0):
    num_periods_per_year = 252  # Assuming 252 trading days in a year
    risk_free_rate_daily = np.power(1 + risk_free_rate, 1 / num_periods_per_year) - 1
    
    returns = np.asarray(returns)
    excess_returns = returns - risk_free_rate_daily
    downside_returns = np.where(excess_returns < target_rate, excess_returns**2, 0)
    downside_deviation = np.sqrt(downside_returns.mean())
    return excess_returns.mean() / downside_deviation


# Define objective function (negative Sharpe Ratio)
def negative_sharpe_ratio(weights, mean_returns, cov_matrix, daily_risk_free_rate):
    portfolio_return = np.dot(weights, mean_returns)
    portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    sharpe_ratio = (portfolio_return - daily_risk_free_rate) / portfolio_volatility
    return -sharpe_ratio

# Define objective function (portfolio volatility)
def portfolio_volatility(weights, cov_matrix):
    return np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))