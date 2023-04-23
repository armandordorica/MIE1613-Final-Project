import pandas as pd 
from Stocks import Stock 
import numpy as np
import pandas as pd
from scipy.stats import norm

class Portfolio:
    def __init__(self, tickers, start_dt, end_dt, timeframe="1D", var_confidence_level=0.95, cvar_confidence_level=0.95):
        self.stocks = {}
        self.summary_statistics = pd.DataFrame(columns=['ticker', 'mean', 'variance', 'skewness', 'kurtosis', 'VaR', 'cVaR'])
        self.pct_changes = pd.DataFrame()
        self.all_stocks_df = pd.DataFrame()

        for ticker in tickers:
            stock = Stock(ticker, start_dt, end_dt, timeframe)
            self.stocks[ticker] = stock

            mean = stock.ticker_data['pct_change'].mean()
            variance = stock.ticker_data['pct_change'].var(ddof=0)
            skewness = stock.ticker_data['skew'].iloc[0]
            kurtosis = stock.ticker_data['kurtosis'].iloc[0]

            # Calculate VaR and cVaR
            var = self.calculate_var(stock.ticker_data['pct_change'].dropna(), var_confidence_level)
            cvar = self.calculate_cvar(stock.ticker_data['pct_change'].dropna(), cvar_confidence_level)

            self.summary_statistics = self.summary_statistics.append({
                'ticker': ticker,
                'mean': mean,
                'variance': variance,
                'skewness': skewness,
                'kurtosis': kurtosis,
                'VaR': var,
                'cVaR': cvar
            }, ignore_index=True)

            self.pct_changes[ticker] = stock.ticker_data['pct_change']
            self.all_stocks_df = self.all_stocks_df.append(stock.ticker_data)

        self.correlation_matrix = self.pct_changes.corr()

    def display_summary_statistics(self):
        print(self.summary_statistics)

    def display_correlation_matrix(self):
        print(self.correlation_matrix)

    def calculate_var(self, returns, confidence_level):
        # Calculate the Value at Risk (VaR) using the parametric method
        mean = returns.mean()
        std_dev = returns.std(ddof=0)
        z_score = norm.ppf(confidence_level)
        var = mean - z_score * std_dev
        return var

    def calculate_cvar(self, returns, confidence_level):
        # Calculate the Conditional Value at Risk (cVaR) using the historical method
        var = self.calculate_var(returns, confidence_level)
        losses_below_var = returns[returns < var]
        cvar = losses_below_var.mean()
        return cvar
