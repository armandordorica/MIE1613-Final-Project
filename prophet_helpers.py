from prophet import Prophet
import pandas as pd
import matplotlib.pyplot as plt

def plot_prediction(input_df, ticker='QQQ', test_period=30): 
    model = Prophet()
    data = input_df.ticker_data[[f'Close']].reset_index().copy(deep=True)
    data = data.rename(columns={'index': 'ds', f'Close': 'y'})

    train_data = data.iloc[0:-test_period]
    model.fit(train_data)

    future = model.make_future_dataframe(periods=test_period)
    forecast = model.predict(future)


    # Preparing data
    observed_data = data.copy()
    observed_data['ds'] = pd.to_datetime(data['ds'])  # Convert to datetime format

    # Extracting forecasted data
    forecasted_data = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]

    forecasted_data = forecasted_data[forecasted_data['ds']>train_data['ds'].max()]
    forecasted_data['ds']= pd.to_datetime(data.iloc[-test_period:]['ds'])
    forecasted_data

    # Plotting the observed data
    plt.figure(figsize=(12, 6))
    plt.plot(observed_data['ds'], observed_data['y'], label='Observed', color='blue')

    # Plotting the forecasted data
    plt.plot(forecasted_data['ds'], forecasted_data['yhat'], label='Forecast', color='red')

    # Plotting the uncertainty intervals
    plt.fill_between(forecasted_data['ds'], forecasted_data['yhat_lower'], forecasted_data['yhat_upper'], color='pink', alpha=0.5)

    # Customize the plot
    plt.xlabel('Date')
    plt.ylabel('Close Price')
    plt.title(f'{ticker} Stock Price Forecast')
    plt.legend()
    plt.grid()

    # Show the plot
    plt.show()
