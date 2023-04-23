import pandas as pd
from datetime import datetime, timedelta

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


