import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt
from data_loader import stock_data_paths


class Predictor:
    def __init__(self):
        self.stocks = stock_data_paths
        self.length = len(self.stocks)
        self.models = [Prophet() for i in range(0, self.length)]
        self.stock_df = [pd.read_csv(self.stocks[i])
                         for i in range(0, self.length)]
        self.future_stock_df = [None for i in range(0, self.length)]

    def train(self, index):
        df = pd.read_csv(self.stocks[index])
        df = df.rename(columns={"Date": "ds", "Close": "y"})
        m = self.models[index]
        m.fit(df)

    def predict(self, index):
        m = self.models[index]
        future = m.make_future_dataframe(periods=10)
        return m.predict(future)

    def iter(self, index):
        self.train(index)
        self.future_stock_df[index] = self.predict(index)

    def run(self):
        for i in range(0, self.length):
            self.iter(i)


p = Predictor()
p.run()
