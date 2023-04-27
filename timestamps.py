import pandas as pd
from datetime import datetime, timedelta
from pandas.tseries.offsets import CustomBusinessDay
from pandas.tseries.holiday import USFederalHolidayCalendar

def subtract_dates(date1, date2):
    date1_obj = datetime.strptime(date1, '%Y-%m-%d')
    date2_obj = datetime.strptime(date2, '%Y-%m-%d')
    diff = date1_obj - date2_obj
    return diff.days

def subtract_days(date_str, days):
    date_obj = datetime.strptime(date_str, '%Y-%m-%d')
    new_date_obj = date_obj - timedelta(days=days)
    return new_date_obj.strftime('%Y-%m-%d')


def subtract_trading_days(date_str, days):
    date_obj = pd.to_datetime(date_str)
    offset = pd.offsets.BDay(n=days * -1)
    new_date_obj = offset.apply(date_obj)
    return new_date_obj.strftime('%Y-%m-%d')



def get_trading_days(start_date, n):
    """
    Generates a list of N trading days (excluding weekends and US federal holidays) starting from a given start date.
    :param start_date: A string or datetime object representing the start date.
    :param n: An integer representing the number of trading days to generate.
    :return: A list of N trading days.
    """
    # Define the US federal holiday calendar
    us_cal = USFederalHolidayCalendar()

    # Define the custom business day offset, excluding weekends and US federal holidays
    us_bd = CustomBusinessDay(calendar=us_cal)

    # Generate a pandas date range of N trading days starting from the given start date
    trading_days = pd.date_range(start=start_date, periods=n, freq=us_bd)

    # Convert the pandas date range to a list of datetime objects
    trading_days = trading_days.to_pydatetime().tolist()

    return trading_days


