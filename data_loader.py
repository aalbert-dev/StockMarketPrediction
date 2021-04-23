import os
from datetime import *

stock_names_path = "/home/arjun/Desktop/Programming Assignments/StockMarketPrediction/15stocks.txt"
stock_data_path = "/home/arjun/Desktop/Programming Assignments/StockMarketPrediction/data"

min_data_length = 90

def get_stock_names():
    with open(stock_names_path) as f:
        return next(f).split(",")


stock_names = get_stock_names()


def get_stock_data(name):
    full_name = os.path.join(stock_data_path, name.strip() + ".csv")
    stock_data = []
    first_row = True
    with open(full_name) as f:
        for row in f:
            if not first_row:
                if len(row) > min_data_length: 
                    contents = row.rstrip().split(",")
                    stock_data.append(get_stock_contents(contents))
            else:
                first_row = False
    return stock_data


def get_stock_contents(sample):
    # Date, Open, High, Low, Close, Volume, Dividends, Stock Splits, Symbol, Sector, Industry
    date = datetime.strptime(sample[0], "%Y-%m-%d")
    open_ = float(sample[1])
    high = float(sample[2])
    low = float(sample[3])
    close = float(sample[4])
    volume = float(sample[5])
    dividends = float(sample[6])
    stock_splits = float(sample[7])
    symbol = str(sample[8])
    sector = str(sample[9])
    industry = str(sample[10])
    return [date, open_, high, low, close, volume, dividends, stock_splits, symbol, sector, industry]


all_stock_data = [get_stock_data(stock_name) for stock_name in stock_names]