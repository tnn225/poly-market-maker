from datetime import datetime, timezone, timedelta
import os
import time
import pandas as pd
import numpy as np
import logging
import json

import csv
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

from poly_market_maker.utils.cache import KeyValueStore
from poly_market_maker.models.binance import Binance
SPREAD = 0.01
DAYS = 20
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
        self.cache = KeyValueStore(db_path="./data/intervals.sqlite")
        # self.read_dates()

    def get_data(self, symbol: str, timestamp: int, only_cache=False):
        """Get target price from Polymarket API."""
        # print(f"Getting data for {symbol} at {timestamp}")
        timestamp = timestamp // 900 * 900

        if self.cache.exists(timestamp):
            return json.loads(self.cache.get(timestamp))
        if only_cache:
            return None

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
                print(f"Getting data from {url}")

                response = requests.get(url)
                response.raise_for_status()
                data = response.json()
                interval_data = {
                    "interval": timestamp,
                    "openPrice": data.get('openPrice'),
                    "closePrice": data.get('closePrice'),
                    "completed": data.get('completed'),
                }
                # print(f"interval_data: {interval_data}")
                if interval_data.get('completed'):
                    # print(f"Caching data for timestamp {timestamp}")
                    self.cache.set(timestamp, json.dumps(interval_data))
                return interval_data
            except Exception as e:
                print(f"Error fetching target for timestamp {timestamp}: {e}")
                time.sleep(10)
        return None 


    def date_to_timestamp(self, date: str) -> int:
        """Convert date string (YYYY-MM-DD) to Unix timestamp (seconds)."""
        return int(datetime.strptime(date, "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp())

    def read_dates(self):
        today = datetime.now()
        dates = [(today - timedelta(days=i+1)).strftime("%Y-%m-%d") for i in range(self.days)]
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
        self.df['delta'] = self.df['closePrice'] - self.df['openPrice']
        self.df['is_up'] = self.df['closePrice'] >= self.df['openPrice']
        return self.df

def show_delta_distribution(df):
    df = df.sort_values('interval')
    plt.figure(figsize=(10, 6))
    plt.hist(df['delta'], bins=100)
    plt.xlabel('Delta')
    plt.ylabel('Frequency')
    plt.title('Distribution of Delta')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show(block=True)  # Keep chart open until manually closed

def show_previous_delta_vs_is_up(df):
    df = df.sort_values('interval')
    df['previous_delta'] = df['delta'].shift(1)
    
    # Remove rows with NaN previous_delta
    df = df.dropna(subset=['previous_delta', 'is_up'])
    
    # Create buckets for previous_delta using custom bins
    bins = [-5000, -1000, -500, 0, 500, 1000,5000] # -1000 to -300, -300 to 0, 0 to 300, 300 to 1000
    df['previous_delta_bucket'] = pd.cut(df['previous_delta'], bins=bins, labels=False, include_lowest=True)
    
    # Calculate mean is_up for each bucket
    bucket_stats = df.groupby('previous_delta_bucket').agg({
        'previous_delta': 'mean',  # Use mean of previous_delta as bucket center
        'is_up': ['mean', 'count']  # Mean and count of is_up
    })
    bucket_stats.columns = ['previous_delta_mean', 'is_up_mean', 'count']
    bucket_stats = bucket_stats.reset_index()
    
    # Remove buckets with too few samples (optional)
    bucket_stats = bucket_stats[bucket_stats['count'] > 0]
    
    # Create subplots: one for mean is_up, one for count
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10), sharex=True)
    
    # Top plot: Dot plot of mean is_up
    ax1.scatter(bucket_stats['previous_delta_mean'], bucket_stats['is_up_mean'], 
                s=50, alpha=0.7, color='blue')
    ax1.set_ylabel('Mean is_up')
    ax1.set_title('Mean is_up by Previous Delta Buckets')
    ax1.grid(True, alpha=0.3)
    
    # Bottom plot: Bar chart of count
    ax2.bar(bucket_stats['previous_delta_mean'], bucket_stats['count'], 
            width=(bucket_stats['previous_delta_mean'].max() - bucket_stats['previous_delta_mean'].min()) / len(bucket_stats) * 0.8,
            alpha=0.7, color='green')
    ax2.set_xlabel('Previous Delta (bucket center)')
    ax2.set_ylabel('Count')
    ax2.set_title('Count by Previous Delta Buckets')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show(block=True)  # Keep chart open until manually closed

def show_delta_vs_previous_delta(df):
    df = df.sort_values('interval')
    df['previous_delta'] = df['delta'].shift(1)
    plt.figure(figsize=(10, 6))
    plt.scatter(df['delta'], df['previous_delta'])
    plt.xlabel('Delta')
    plt.ylabel('Previous Delta')
    plt.title('Delta vs Previous Delta')
    plt.show(block=True)

def view_delta_binance_and_intervals(binance_df, intervals_df):
    """Match intervals with Binance data and plot delta vs binance_delta"""
    binance_df = binance_df.sort_values('open_time').copy()
    intervals_df = intervals_df.sort_values('interval').copy()
    
    # Calculate deltas
    binance_df['delta'] = binance_df['close'] - binance_df['open']
    intervals_df['delta'] = intervals_df['closePrice'] - intervals_df['openPrice']
    
    # Create a mapping from open_time to delta in binance_df
    binance_dict = dict(zip(binance_df['open_time'], binance_df['delta']))
    
    # Match intervals_df['interval'] with binance_df['open_time'] and add binance_delta
    intervals_df['binance_delta'] = intervals_df['interval'].map(binance_dict)
    
    # Remove rows where we couldn't find a match
    intervals_df = intervals_df.dropna(subset=['delta', 'binance_delta'])
    
    # Plot delta vs binance_delta
    plt.figure(figsize=(10, 6))
    plt.scatter(intervals_df['delta'], intervals_df['binance_delta'], alpha=0.5, s=20)
    plt.xlabel('Intervals Delta')
    plt.ylabel('Binance Delta')
    plt.title('Intervals Delta vs Binance Delta')
    plt.grid(True, alpha=0.3)
    
    # Add diagonal line for reference
    min_val = min(intervals_df['delta'].min(), intervals_df['binance_delta'].min())
    max_val = max(intervals_df['delta'].max(), intervals_df['binance_delta'].max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.5, label='y=x')
    plt.legend()
    
    plt.tight_layout()
    plt.show(block=True)
    
    return binance_df, intervals_df

def main():
    intervals = Interval()
    intervals_df = intervals.df

    binance = Binance(symbol="BTCUSDT", interval="15m")
    binance_df = binance.df

    view_delta_binance_and_intervals(binance_df, intervals_df)
    
    #show_delta_vs_previous_delta(intervals_df)
    #show_previous_delta_vs_is_up(intervals_df)

if __name__ == "__main__":
    main()