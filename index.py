import pandas as pd
import numpy as np
from prophet import Prophet
import matplotlib.pyplot as plt

data = pd.read_csv('dataset.csv')

timestamps = data['timestamp'][0].split(' ')

timestamps = [_timestamp.replace('[', '') for _timestamp in timestamps]
timestamps = [_timestamp.replace(']', '') for _timestamp in timestamps]
timestamps = [_timestamp.strip() for _timestamp in timestamps]
timestamps = [_timestamp.replace('\'', '') for _timestamp in timestamps]
timestamps = [pd.to_datetime(_timestamp) for _timestamp in timestamps]

temperatures = data['t_max'][0].split(' ')
temperatures = [_temperature.replace('[', '') for _temperature in temperatures]
temperatures = [_temperature.replace(']', '') for _temperature in temperatures]
temperatures = [_temperature.strip() for _temperature in temperatures]
temperatures = [_temperature.replace('\'', '') for _temperature in temperatures]
temperatures = list(filter(None, temperatures))

df = pd.DataFrame({'ds': timestamps, 'y': temperatures})

model = Prophet()
model.fit(df)

future = model.make_future_dataframe(periods=2)
forecast = model.predict(future)

def mean_absolute_scaled_error(y_true, y_pred, y_train):
    mae = np.mean(np.abs(y_true - y_pred))
    naive_forecast = np.roll(y_train, 1)
    naive_mae = np.mean(np.abs(y_train[1:] - naive_forecast[1:]))
    mase = mae / naive_mae
    return mase



# # Example usage (replace y_true and y_pred with your actual values)
y_true = df['y'][-10:]  # Adjust this based on your dataset
y_pred = forecast['yhat'][-2:]  # Adjust this based on your forecast
y_train = df['y'][:-10]  # Adjust this based on your dataset

y_true = [float(val) for val in y_true]
y_pred = [float(val) for val in y_pred]
y_train = [float(val) for val in y_train]

y_true = pd.Series(y_true)
y_pred = pd.Series(y_pred)
y_train = pd.Series(y_train)

mase = mean_absolute_scaled_error(y_true, y_pred, y_train)
print('MASE:',mase)
