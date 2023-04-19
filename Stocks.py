# Import libraries and dependencies
import os
import numpy as np
import pandas as pd
import alpaca_trade_api as tradeapi
from MCForecastTools import MCSimulation
from scipy.stats import skew, kurtosis, chi2


import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from pandas_datareader import data as pdr

import yfinance as yf

from dotenv import load_dotenv
from dotenv import dotenv_values



# Load .env enviroment variables


load_dotenv()
config = dotenv_values(".env")

api = tradeapi.REST(
    config['ALPACA_API_KEY'],
    config['ALPACA_SECRET_KEY'],
    api_version = "v2"
)


class Stock: 
    def __init__(self,ticker,start_dt,end_dt, timeframe = "1D"):
        
        self.ticker=ticker
        self.start_dt =start_dt
        self.end_dt = end_dt
        self.timeframe = timeframe
        yf.pdr_override()

        try:    
            print("data from alpaca")
            self.ticker_data = self.get_ticker_data_alpaca()    
        except: 
            print("Data from alpaca not available, pulling from yahoo")
            self.ticker_data = self.get_data_yahoo()
        
        self.ticker_data['ticker'] = ticker
        self.ticker_data['pct_change'] = self.ticker_data['Close'].pct_change()
        self.ticker_data = self.cumulative_percentage_change()
        self.ticker_data = self.normalize_close()
        self.ticker_data = self.cumulative_mean()
        self.ticker_data =self.cumulative_variance()
        self.ticker_data =self.cumulative_std()
        self.ticker_data =self.skew()
        self.ticker_data =self.kurtosis()
        self.ticker_data =self.jarque_bera_statistic()
        self.ticker_data =self.jarque_bera_p_value()
    
    def get_ticker_data_alpaca(self):
         
        ticker_data = api.get_bars(
        self.ticker,
        self.timeframe,
        start=self.start_dt,
        end=self.end_dt,
        limit=1000,
        ).df
        
        self.available_dates = [x.strftime('%Y-%m-%d') for x in ticker_data.index]
        
        self.first_min_date =[x for x in self.available_dates if x>=self.start_dt][0]
        self.first_max_date =[x for x in self.available_dates if x<=self.end_dt][-1]
        
        ticker_data = ticker_data.loc[self.first_min_date: self.first_max_date]
        ticker_data = ticker_data[['open','high', 'low', 'close', 'volume']]
        ticker_data.columns = ['Open','High', 'Low', 'Close', 'Volume']
        
        ticker_data.index = [x.strftime('%Y-%m-%d') for x in ticker_data.index]
        return ticker_data 

        
    def get_data_yahoo(self): 
        data = pdr.get_data_yahoo(self.ticker,
            start=self.start_dt, end=self.end_dt)
        data.drop(columns=['Adj Close'], inplace=True)
        data.index = [x.strftime('%Y-%m-%d') for x in data.index]
        
        self.ticker = yf.Ticker(self.ticker)

        return data
            
        

    def cumulative_percentage_change(self):
        if 'Close' not in self.ticker_data.columns:
            raise ValueError("DataFrame must have a 'Close' column")

        initial_close = self.ticker_data['Close'].iloc[0]
        self.ticker_data['Cumulative_Percentage_Change'] = (self.ticker_data['Close'] / initial_close - 1) * 100

        return self.ticker_data
    
    
    def normalize_close(self):
        if 'Close' not in self.ticker_data.columns:
            raise ValueError("DataFrame must have a 'Close' column")

        min_close = self.ticker_data['Close'].min()
        max_close = self.ticker_data['Close'].max()
        self.ticker_data['normalized_close'] = (self.ticker_data['Close'] - min_close) / (max_close - min_close)

        return self.ticker_data
    

    def cumulative_mean(self):
        if 'normalized_close' not in self.ticker_data.columns:
            raise ValueError("DataFrame must have a 'normalized_close' column")

        self.ticker_data['cumulative_mean_normalized_close'] = self.ticker_data['normalized_close'].expanding().mean()

        return self.ticker_data

    def cumulative_variance(self):
        if 'normalized_close' not in self.ticker_data.columns:
            raise ValueError("DataFrame must have a 'normalized_close' column")

        self.ticker_data['cumulative_variance_normalized_close'] = self.ticker_data['normalized_close'].expanding().var(ddof=0)

        return self.ticker_data

    def cumulative_std(self):
        if 'normalized_close' not in self.ticker_data.columns:
            raise ValueError("DataFrame must have a 'normalized_close' column")

        self.ticker_data['cumulative_std_normalized_close'] = self.ticker_data['normalized_close'].expanding().std(ddof=0)

        return self.ticker_data
    
    
    def skew(self):
        if 'pct_change' not in self.ticker_data.columns:
            raise ValueError("DataFrame must have a 'pct_change' column")

        self.ticker_data['skew'] = skew(self.ticker_data['pct_change'].iloc[1:])

        return self.ticker_data
    
    
    def kurtosis(self):
        if 'pct_change' not in self.ticker_data.columns:
            raise ValueError("DataFrame must have a 'pct_change' column")

        self.ticker_data['kurtosis'] = kurtosis(self.ticker_data['pct_change'].iloc[1:])

        return self.ticker_data
    
    
    def jarque_bera_statistic(self):
        
        if 'pct_change' not in self.ticker_data.columns:
            raise ValueError("DataFrame must have a 'pct_change' column")
        # Calculate the number of observations (n)
        n = len(self.ticker_data)

        # Calculate the sample skewness (S) and kurtosis (K)
        S = skew(self.ticker_data['pct_change'].iloc[1:])
        K = kurtosis(self.ticker_data['pct_change'].iloc[1:], fisher=False)

        # Compute the Jarque-Bera statistic
        JB = n * (S**2 / 6 + (K - 3)**2 / 24)
        
        self.ticker_data['Jarque_Bera_stat'] = JB

        return self.ticker_data
    
    def jarque_bera_p_value(self):
        
        if 'pct_change' not in self.ticker_data.columns:
            raise ValueError("DataFrame must have a 'pct_change' column")
        # Calculate the number of observations (n)
        n = len(self.ticker_data)

        # Calculate the sample skewness (S) and kurtosis (K)
        S = skew(self.ticker_data['pct_change'].iloc[1:])
        K = kurtosis(self.ticker_data['pct_change'].iloc[1:], fisher=False)

        # Compute the Jarque-Bera statistic
        JB = n * (S**2 / 6 + (K - 3)**2 / 24)
        
        p_value = chi2.sf(JB, 2)
        self.ticker_data['Jarque_Bera_p_val'] = p_value

        return self.ticker_data
    
    
