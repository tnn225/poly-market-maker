from datetime import datetime, timezone
import time
import pandas as pd
import numpy as np

import csv

from collections import deque

from scipy.stats import norm

VOL_WINDOW = 10000

class PricePrediction:
    def __init__(self, timestamp: int):
        self.target = None
        self.timestamp = timestamp
        self.sigma = 0.0
        self.mu = 0.0
        self.prices = deque(maxlen=VOL_WINDOW)  # last N prices 
        now = int(time.time())
        date = datetime.fromtimestamp(now, tz=timezone.utc).strftime("%Y-%m-%d")
        filename = f'./data/price_{date}.csv'
        self.read_prices(filename)
        self.sigma = self.estimate_sigma()

    def set_timestamp(self, timestamp, price):
        self.timestamp = timestamp
        self.target = price

    def read_prices(self, filename='./data/price_2025-11-17.csv'):
        with open(filename, mode='r', newline='') as file:
            csv_reader = csv.reader(file)

            # Optionally, skip the header row if present
            header = next(csv_reader)
            print(f"Header: {header}")

            # Iterate and print each data row
            for row in csv_reader:
                # print(f"Data Row: {row}")
                timestamp = int(row[0])
                price = float(row[1])  
                self.prices.append(price)

                if timestamp == self.timestamp:
                    self.target = price
                    print(f"Set target price to {self.target} at timestamp {timestamp}")     

    def estimate_sigma(self):
        if len(self.prices) < 2:
            return 0.0
        # print(self.prices)
        log_returns = np.diff(np.log(self.prices))
        return np.std(log_returns)

    def get_probability(self, price, seconds_left):
        if self.target is None:
            return None 

        # print(price, target, seconds_left)
        if seconds_left <= 0 or self.sigma == 0:
            return float(price >= self.target)
        
        z = (np.log(price / self.target) - (self.mu - 0.5 * self.sigma**2) * seconds_left) / (self.sigma * np.sqrt(seconds_left))
        return norm.cdf(z)
    
    def add_price(self, timestamp, price):
        # print(f"Add price to {price} at timestamp {timestamp}")
        self.prices.append(price)
        self.sigma = self.estimate_sigma()
        if timestamp % 900 == 0:
            print(f"Updating target price to {price} at timestamp {timestamp}")
            self.timestamp = timestamp
            self.target = price     
