from lstm_sample import LSTM
from dataset import StockMarketDataset

network = LSTM()

data = StockMarketDataset()

epochs = 50

for epoch in range(0, epochs):
    # date, symbol, industry, sector
    # vectorize(date, symbol, industry, sector)
    # train
    # see results for future date
    pass