import plotly.graph_objects as go
import numpy as np
from scipy.optimize import minimize
import financial_ratios as fr

# Minimize negative Sharpe Ratio subject to constraints
def maximize_sharpe_ratio(mean_returns, cov_matrix, daily_risk_free_rate, num_stocks, min_weight, max_weight):
    cons = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bounds = [(min_weight, max_weight)] * num_stocks
    init_guess = [1 / num_stocks] * num_stocks

    result = minimize(fr.negative_sharpe_ratio, init_guess, args=(mean_returns, cov_matrix, daily_risk_free_rate), bounds=bounds, constraints=cons)
    return result


# Minimize volatility subject to constraints
def minimize_volatility(cov_matrix, target_return, num_stocks, min_weight, max_weight):
    cons = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1},
            {'type': 'eq', 'fun': lambda x: np.dot(x, mean_returns) - target_return})
    bounds = [(min_weight, max_weight)] * num_stocks
    init_guess = [1 / num_stocks] * num_stocks

    result = minimize(fr.portfolio_volatility, init_guess, args=(cov_matrix), bounds=bounds, constraints=cons)
    return result
