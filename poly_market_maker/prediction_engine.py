from datetime import datetime, timezone
import os
import time
import pandas as pd
import numpy as np

import csv

from collections import deque

from scipy.stats import norm

VOL_WINDOW = 10000

class PricePrediction:
    def __init__(self):
        self.sigma = 0.0
        self.mu = 0.0
        self.prices = deque(maxlen=VOL_WINDOW)  # last N prices 
        date = datetime.fromtimestamp(int(time.time()), tz=timezone.utc).strftime("%Y-%m-%d")
        filename = f'./data/price_{date}.csv'
        if os.path.exists(filename):
            self.read_prices(filename)
            self.sigma = self.estimate_sigma()

    def read_prices(self, filename):
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

    def estimate_sigma(self):
        if len(self.prices) < 2:
            return 0.0
        # print(self.prices)
        log_returns = np.diff(np.log(self.prices))
        return np.std(log_returns)

    def get_probability(self, price, target, seconds_left):
        # print(price, target, seconds_left)
        if seconds_left <= 0 or self.sigma == 0:
            return float(price >= target)
        
        z = (np.log(price / target) - (self.mu - 0.5 * self.sigma**2) * seconds_left) / (self.sigma * np.sqrt(seconds_left))
        return norm.cdf(z)
