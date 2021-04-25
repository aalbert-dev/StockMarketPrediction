import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt

df = pd.read_csv("/home/arjun/Desktop/Programming Assignments/StockMarketPrediction/data/TRI.csv")
df = df.rename(columns={"Date": "ds", "Close": "y"})
print(df.head())

m = Prophet()
m.fit(df)

future = m.make_future_dataframe(periods=365)
print(future.tail())

forecast = m.predict(future)
print(forecast.columns)
print(forecast.tail())
# forecast[['ds', 'y']].tail()