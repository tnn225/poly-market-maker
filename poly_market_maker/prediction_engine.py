from datetime import datetime, timezone, timedelta
import os
import time
import pandas as pd
import numpy as np
import logging
import random
import csv

from poly_market_maker.dataset import Dataset
from collections import deque

from scipy.stats import norm

WINDOW = 900
logger = logging.getLogger(__name__)

SECONDS_PER_YEAR = 365.0 * 24.0 * 60.0 * 60.0

class PredictionEngine:
    def __init__(self):
        self.prices = deque(maxlen=WINDOW)

    def add_price(self, price):
        self.prices.append(price)

    def get_sigma(self):
        # `self.prices` is a deque, so compute log returns manually.
        if len(self.prices) < 2:
            return 0.0
        prices = np.asarray(self.prices, dtype=float)
        # Filter non-positive prices to avoid log/zero issues.
        if np.any(prices <= 0):
            return 0.0

        log_returns = np.log(prices[1:] / prices[:-1])
        sigma = float(np.std(log_returns, ddof=0))
        return float(sigma) #  * np.sqrt(SECONDS_PER_YEAR))

    def get_probability(self, price, target, seconds_left):
        sigma = self.get_sigma()
        # Remaining-time vol using annualized sigma:
        # sigma_rem = sigma_annual * sqrt(seconds_left / seconds_per_year)
        time_factor = np.sqrt(float(seconds_left)) # / np.sqrt(SECONDS_PER_YEAR)
        sigma = sigma * time_factor
        if sigma == 0:
            return price >= target
        if seconds_left <= 0:
            return price >= target
        log_return = np.log(price / target) 
        z_score = log_return / sigma
        probability = norm.cdf(z_score)
        return probability
    

def main():
    dataset = Dataset(days=1)
    df = dataset.df
    prediction_engine = PredictionEngine()

    for index, row in df.iterrows():
        price = row['price']
        target = row['target']
        bid = row['bid']
        ask = row['ask']
        seconds_left = row['seconds_left']
        prediction_engine.add_price(price)

        delta = price - target
        up = prediction_engine.get_probability(price, target, seconds_left)  
        sigma = prediction_engine.get_sigma()
        print(f"{price} {delta:+.4f} sigma: {sigma:.6f} bid: {bid:.2f} up: {up:.2f}")
    

if __name__ == "__main__":
    main()