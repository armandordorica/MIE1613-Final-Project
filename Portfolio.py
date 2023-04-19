import pandas as pd 
from Stocks import Stock 

class Portfolio:
    def __init__(self, tickers, start_dt, end_dt, timeframe="1D"):
        self.stocks = {}
        self.summary_statistics = pd.DataFrame(columns=['ticker', 'mean', 'variance', 'skewness', 'kurtosis'])
        self.pct_changes = pd.DataFrame()

        for ticker in tickers:
            stock = Stock(ticker, start_dt, end_dt, timeframe)
            self.stocks[ticker] = stock

            mean = stock.ticker_data['pct_change'].mean()
            variance = stock.ticker_data['pct_change'].var(ddof=0)
            skewness = stock.ticker_data['skew'].iloc[0]
            kurtosis = stock.ticker_data['kurtosis'].iloc[0]

            self.summary_statistics = self.summary_statistics.append({
                'ticker': ticker,
                'mean': mean,
                'variance': variance,
                'skewness': skewness,
                'kurtosis': kurtosis
            }, ignore_index=True)

            self.pct_changes[ticker] = stock.ticker_data['pct_change']

        self.correlation_matrix = self.pct_changes.corr()

    def display_summary_statistics(self):
        print(self.summary_statistics)

    def display_correlation_matrix(self):
        print(self.correlation_matrix)
