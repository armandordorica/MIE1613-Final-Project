import numpy as np

def sharpe_ratio(returns, risk_free_rate=0.02):
    """
    Calculate the Sharpe Ratio.

    Args:
    returns (list, numpy.array, or pandas.Series): Returns for a financial instrument.
    risk_free_rate (float, optional): The risk-free rate, default is 0.02 (2%).

    Returns:
    float: The Sharpe Ratio.
    """
    returns = np.asarray(returns)
    excess_returns = returns - risk_free_rate
    return excess_returns.mean() / excess_returns.std()


def sortino_ratio(returns, target_rate=0.0, risk_free_rate=0.02):
    """
    Calculate the Sortino Ratio.

    Args:
    returns (list, numpy.array, or pandas.Series): Returns for a financial instrument.
    target_rate (float, optional): The target or required rate of return, default is 0.0.
    risk_free_rate (float, optional): The risk-free rate, default is 0.02 (2%).

    Returns:
    float: The Sortino Ratio.
    """
    returns = np.asarray(returns)
    excess_returns = returns - risk_free_rate
    downside_risk = np.sqrt(np.mean(np.minimum(excess_returns - target_rate, 0.0) ** 2))
    return excess_returns.mean() / downside_risk
