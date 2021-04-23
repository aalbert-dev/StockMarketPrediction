from torch.utils.data import Dataset
from data_loader import all_stock_data
import numpy as np

class StockMarketDataset(Dataset):

    def __init__(self):
        self.raw_data = all_stock_data
        self.length = len(self.raw_data)

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        return self.raw_data[index]
