from datetime import datetime, timezone, timedelta
import os
import time
import pandas as pd
import numpy as np
import logging

import csv
import requests_cache
import requests
from collections import deque

from lightgbm import LGBMClassifier

from scipy.stats import norm
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    confusion_matrix,
    roc_auc_score,
)


import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt

SPREAD = 0.01
DAYS = 30
SECONDS_LEFT_BIN_SIZE = 15
SECONDS_LEFT_BINS = int(900 / SECONDS_LEFT_BIN_SIZE)

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s"
)


FEATURE_COLS = ['interval', 'openPrice', 'closePrice']

class Interval:
    def __init__(self, days=DAYS):
        self.days = days
        self.session = requests_cache.CachedSession("./data/intervals.sqlite", backend="sqlite", expire_after=86400)

        self._read_dates()

    def get_data(self, symbol: str, timestamp: int):
        """Get target price from Polymarket API with DataFrame caching."""
        # Fetch from API
        eventStartTime = datetime.fromtimestamp(timestamp, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        endDate = datetime.fromtimestamp(timestamp + 900, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        params = {
            "symbol": symbol,
            "eventStartTime": eventStartTime,
            "variant": "fifteen",
            "endDate": endDate
        }
        
        url = requests.Request('GET', "https://polymarket.com/api/crypto/crypto-price", params=params).prepare().url
        
        for i in range(3):
            try:
                if self.session.cache.contains(url=url) is False:
                    return None

                response = self.session.get(url)
                response.raise_for_status()
                data = response.json()
                interval_data = {
                    "interval": timestamp,
                    "openPrice": data.get('openPrice'),
                    "closePrice": data.get('closePrice'),
                    "completed": data.get('completed'),
                }
                # print(f"interval_data: {interval_data}")
                if interval_data.get('completed') is False:
                    self.session.cache.delete(url)
                return interval_data
            except Exception as e:
                print(f"Error fetching target for timestamp {timestamp}: {e}")
                time.sleep(10)
        return None 


    def date_to_timestamp(self, date: str) -> int:
        """Convert date string (YYYY-MM-DD) to Unix timestamp (seconds)."""
        return int(datetime.strptime(date, "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp())

    def _read_dates(self):
        today = datetime.now()
        dates = [(today - timedelta(days=i)).strftime("%Y-%m-%d") for i in range(self.days)]
        # Load data from CSV files for each date
        dataframes = []
        for date in dates:
            timestamp = self.date_to_timestamp(date)
            for i in range(96):
                data = self.get_data("BTC", timestamp + i * 900)
                if data is not None:
                    row = [data.get('interval'), data.get('openPrice'), data.get('closePrice')]
                    dataframes.append(row)
        self.df = pd.DataFrame(dataframes, columns=FEATURE_COLS)
        return self.df

def main():
    intervals = Interval()
    df = intervals.df
    df['delta'] = df['closePrice'] - df['openPrice']
    
    # Also plot histogram with frequency as percentage using pd.qcut
    plt.figure(figsize=(10, 6))
    delta_values = df['delta'].dropna()
    
    # Create quantile-based bins
    n_bins = 100
    delta_cut = pd.qcut(delta_values, q=n_bins, duplicates='drop')
    
    # Count values in each bin and calculate percentages
    bin_counts = delta_cut.value_counts().sort_index()
    bin_percentages = (bin_counts / len(delta_values) * 100).values
    
    # Get bin edges for x-axis
    bin_edges = [interval.left for interval in bin_counts.index] + [bin_counts.index[-1].right]
    bin_centers = [(bin_edges[i] + bin_edges[i+1]) / 2 for i in range(len(bin_edges) - 1)]
    
    plt.bar(bin_centers, bin_percentages, width=[bin_edges[i+1] - bin_edges[i] for i in range(len(bin_edges) - 1)], 
            edgecolor='black', alpha=0.7)
    plt.xlabel('Delta (closePrice - openPrice)')
    plt.ylabel('Frequency (%)')
    plt.title('Distribution of Delta (Quantile-based Bins)')
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.show()

    return 
if __name__ == "__main__":
    main()