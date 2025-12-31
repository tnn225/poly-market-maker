from datetime import datetime, timezone, timedelta
import os
import time
import pandas as pd
import numpy as np
import logging

import csv

from collections import deque

from lightgbm import LGBMClassifier
from poly_market_maker.intervals import Interval

from scipy.stats import norm
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    confusion_matrix,
    roc_auc_score,
)

from poly_market_maker.models.binance import Binance

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


FEATURE_COLS = ['seconds_left_log', 'log_return', 'delta', 'seconds_left', 'bid', 'ask', 'z_score', 'prob_est']

class Dataset:
    def __init__(self, days=DAYS):
        self.days = days
        self.intervals = Interval(DAYS)
        self._read_dates()
        self._add_target_and_is_up()
        self._train_test_split()
        self.feature_cols = FEATURE_COLS
        self.show()


    def show(self):
        print(self.train_df.head())
        print(self.test_df.head())

        print("Train/Test set overview:")
        print(f"  train_df shape: {self.train_df.shape}")
        print(f"  test_df shape:  {self.test_df.shape}")
        print(f"  train labels distribution:\n {self.train_df['label'].value_counts(normalize=True)} {self.train_df['label'].value_counts(normalize=False)}")
        print(f"  test labels distribution:\n {self.test_df['label'].value_counts(normalize=True)} {self.test_df['label'].value_counts(normalize=False)}\n")

    def _read_dates(self):
        today = datetime.now()
        # dates = [(today - timedelta(days=i)).strftime("%Y-%m-%d") for i in range(self.days)]
        dates = ["2025-12-16"]
        # Load data from CSV files for each date
        dataframes = []
        for date in dates:
            if os.path.exists(f"./data/prices/price_{date}.csv"):
                df = self._read_rows(date)
                print(f"date: {date}, df shape: {df.shape}")
                if df is not None and not df.empty:
                    dataframes.append(df)

        # Combine all dataframes
        if dataframes:
            self.df = pd.concat(dataframes, ignore_index=True)
        else:
            self.df = pd.DataFrame(columns=["timestamp", "price", "bid", "ask"])

        self.df['timestamp'] = self.df['timestamp'].astype(int)
        self.df['price'] = self.df['price'].astype(float)
        self.df['bid'] = round(self.df['bid'].astype(float), 2)
        self.df['ask'] = round(self.df['ask'].astype(float), 2) 


    def _read_rows(self, date):
        path = f"./data/prices/price_{date}.csv"
        if not os.path.exists(path):
            logger.warning(f"File not found: {path}")
            return None
        
        try:
            df = pd.read_csv(path)
            return df
        except Exception as e:
            logger.error(f"Error reading {path}: {e}")
            return None

    def _balance_df(self, df):
        grouped = df.groupby('label')['timestamp'].unique()
        #print(grouped)
        for label, timestamps in grouped.items():
            # print(label, timestamps)
            timestamps = sorted(timestamps)
            print(f"initial label {label}, len: {len(timestamps)}")

        # Find minimum count of unique timestamps across all labels
        min_timestamps_count = min(len(timestamps) for timestamps in grouped)
        # print(f"Minimum timestamp count: {min_timestamp_count}")

        for label, timestamps in grouped.items():
            # print(label, timestamps)
            timestamps = sorted(timestamps)
            if len(timestamps) > min_timestamps_count:
                timestamp = timestamps[min_timestamps_count]
                # print(f"timestamp: {timestamp}")
                df = df[(df['timestamp'] < timestamp) | (df['label'] != label)]
        #print(df.head())
        grouped = df.groupby('label')['timestamp'].unique()
        for label, timestamps in grouped.items():
            # print(label, timestamps)
            timestamps = sorted(timestamps)
            print(f"final label {label}, len: {len(timestamps)}")

        return df

    def _add_target_and_is_up(self):
        # Initialize columns
        self.df['target'] = None
        self.df['is_up'] = None
        
        # Calculate 15-minute interval start time (rounded down to nearest 900 seconds)
        self.df['interval'] = self.df['timestamp'] // 900 * 900
        
        # Create price lookup dictionary for O(1) access
        price_dict = dict(zip(self.df['timestamp'], self.df['price']))
        for interval in self.df['interval'].unique():
            data = self.intervals.get_data('BTC', interval, only_cache=True)
            if data:
                price_dict[interval] = data['openPrice']
                price_dict[interval + 900] = data['closePrice']
        
        # Calculate target and is_up for each unique interval
        interval_values = {}
        for interval in self.df['interval'].unique():
            target = price_dict.get(interval)
            
            if target is not None:
                # Get price at the end of the interval (interval + 900 seconds)
                next_interval_price = price_dict.get(interval + 900)
                
                if next_interval_price is not None:
                    # is_up = True if price goes up or stays same
                    interval_values[interval] = {
                        'target': target,
                        'is_up': target <= next_interval_price
                    }
        
        # Apply the same target and is_up to all rows in [interval, interval+900)
        for interval, values in interval_values.items():
            mask = (self.df['interval'] == interval)
            self.df.loc[mask, 'target'] = values['target']
            self.df.loc[mask, 'is_up'] = values['is_up']

        # Filter out rows where target or is_up couldn't be calculated
        self.df = self.df[
            (self.df['is_up'].notna())
            & (self.df['bid'].notna())
            & (self.df['ask'].notna())
            & (self.df['target'].notna())
            & ((self.df['bid'] > 0) | (self.df['ask'] > 0))
            & (self.df['price'] > 0)
            & (self.df['target'] > 0)
        ].copy()

        price_vals = self.df['price'].astype(float).to_numpy()
        target_vals = self.df['target'].astype(float).to_numpy()
        self.df["delta"] = price_vals - target_vals
        self.df["percent"] = (price_vals - target_vals) / target_vals
        self.df["log_return"] = np.log(price_vals / target_vals)


        # self.df = self.df[
        #     (self.df['delta'] > -150)
        #     & (self.df['delta'] < 150)
        # ].copy()
    
        self.df['seconds_left'] = 900 - (self.df['timestamp'] - self.df['interval'])
        self.df['time'] = self.df['seconds_left'].astype(float) / 900.
        self.df['seconds_left_log'] = np.log(self.df['seconds_left'])
                
        # Sort by timestamp to ensure proper rolling window calculation
        self.df = self.df.sort_values('timestamp').reset_index(drop=True)
        
        # Compute rolling sigma of log_return with window 900
        self.df['sigma'] = self.df['log_return'].rolling(window=60, min_periods=1).std()
        
        # Estimate z_score from log_return, sigma, and seconds_left
        # z_score = log_return / (sigma * sqrt(seconds_left / 900))
        # This scales the volatility by the time remaining
        time_factor = np.sqrt(self.df['seconds_left'] / 60.0)
        # Avoid division by zero
        sigma_scaled = self.df['sigma'].replace(0, np.nan)
        self.df['z_score'] = self.df['log_return'] / (sigma_scaled * time_factor)
        # Fill NaN values (where sigma was 0) with 0
        self.df['z_score'] = self.df['z_score'].fillna(0)
        
        # Use norm.cdf to get probability estimates
        self.df["prob_est"] = norm.cdf(self.df["z_score"])

        self.df['bid_bin'] = self.df['bid'].apply(self.get_bid_bin)
        self.df['seconds_left_bin'] = self.df['seconds_left'].apply(self.get_seconds_left_bin)

    def get_seconds_left_bin(self, seconds_left):
        return int(seconds_left // SECONDS_LEFT_BIN_SIZE)

    def get_bid_bin(self, bid):
        return int(bid * 100)


    def _train_test_split(self, test_ratio: float = 0.2):
        df_sorted = self.df.sort_values('timestamp').reset_index(drop=True)
        split_idx = int(len(df_sorted) * (1 - test_ratio))
        split_idx = max(1, min(split_idx, len(df_sorted) - 1))
        self.train_df = df_sorted.iloc[:split_idx].copy()
        self.test_df = df_sorted.iloc[split_idx:].copy()

        self.df['label'] = self.df['is_up'].astype(int)
        self.train_df['label'] = self.train_df['is_up'].astype(int)
        self.test_df['label'] = self.test_df['is_up'].astype(int)

        # print(self.train_df.head())
        # print(self.test_df.head())

        self.train_df = self._balance_df(self.train_df)
        self.test_df = self._balance_df(self.test_df)

        return self.train_df, self.test_df

    def evaluate_model_metrics(self, df: pd.DataFrame, probability_column: str = 'probability', spread: float = 0.05):
        eval_df = df.dropna(subset=[probability_column, 'label']).copy()
        y_true = eval_df['label'].astype(int)
        probs = eval_df[probability_column].astype(float)
        y_pred = (probs - spread > eval_df['bid']).astype(int)

        # Calculate confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        # Confusion matrix format: [[TN, FP], [FN, TP]]
        # Handle different matrix sizes
        if cm.shape == (2, 2):
            TN, FP, FN, TP = cm.ravel()
        elif cm.shape == (1, 1):
            # Only one class present in predictions
            if y_pred[0] == 0:
                TN, FP, FN, TP = cm[0, 0], 0, 0, 0
            else:
                TN, FP, FN, TP = 0, 0, 0, cm[0, 0]
        else:
            # Fallback: try to extract values safely
            TN = cm[0, 0] if cm.shape[0] >= 1 and cm.shape[1] >= 1 else 0
            FP = cm[0, 1] if cm.shape[0] >= 1 and cm.shape[1] >= 2 else 0
            FN = cm[1, 0] if cm.shape[0] >= 2 and cm.shape[1] >= 1 else 0
            TP = cm[1, 1] if cm.shape[0] >= 2 and cm.shape[1] >= 2 else 0
        
        # Calculate metrics from confusion matrix
        total = TN + FP + FN + TP
        accuracy = (TN + TP) / total if total > 0 else 0.0
        precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        pnl = float('nan')
        num_trades = 0
        if {'bid', 'is_up'}.issubset(eval_df.columns):
            actions = (probs - spread > eval_df['bid'])
            trade_df = eval_df[actions].copy()
            if len(trade_df) > 0:
                trade_df['revenue'] = trade_df['is_up'].astype(float)
                trade_df['cost'] = trade_df['bid']
                pnl = float((trade_df['revenue'] - trade_df['cost']).sum())
                num_trades = len(trade_df)

        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'roc_auc': roc_auc_score(y_true, probs) if len(np.unique(y_true)) > 1 else float('nan'),
            'total_pnl': pnl,
            'num_trades': num_trades,
            'num_rows': len(eval_df),
            'confusion_matrix': cm.tolist(),
        }

def view_delta_binance_and_dataset(binance_df, dataset_df):
    """Match Binance and dataset by timestamp and plot delta vs delta"""
    binance_df = binance_df.sort_values('open_time').copy()
    print(f"binance_df: {len(binance_df)}")
    dataset_df = dataset_df.sort_values('timestamp').copy()
    print(f"dataset_df: {len(dataset_df)}")
    
    # Create a mapping from open_time to delta in binance_df
    binance_dict = dict(zip(binance_df['open_time'], binance_df['delta']))
    print(f"binance_dict: {len(binance_dict)}")
    
    # Match dataset_df['timestamp'] with binance_df['open_time'] and add binance_delta
    dataset_df['binance_delta'] = dataset_df['timestamp'].map(binance_dict)
    
    # Remove rows where we couldn't find a match
    dataset_df = dataset_df.dropna(subset=['delta', 'binance_delta'])
    print(f"matched rows: {len(dataset_df)}")
    
    # Find indices where binance_delta < -50 or > 50
    extreme_indices = dataset_df[(dataset_df['binance_delta'] < -50) | (dataset_df['binance_delta'] > 50)].index.tolist()
    print(f"found {len(extreme_indices)} extreme points (binance_delta < -50 or > 50)")
    
    # Only show the first extreme range (100 points around first extreme point)
    if len(extreme_indices) > 0:
        first_extreme_idx = extreme_indices[0]
        pos = dataset_df.index.get_loc(first_extreme_idx)
        start = max(0, pos - 50)
        end = min(len(dataset_df), pos + 50)
        dataset_df = dataset_df.iloc[start:end].sort_values('timestamp')
        print(f"showing {len(dataset_df)} points around first extreme point at position {pos}")
    else:
        print("no extreme points found, showing first 300 points")
        if len(dataset_df) > 300:
            dataset_df = dataset_df.head(300)
    
    # Create subplots: delta plot on top, bid plot on bottom
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    
    # Top plot: timestamp on x-axis with delta and binance_delta as two lines
    ax1.plot(dataset_df['timestamp'], dataset_df['delta'], alpha=0.7, label='Dataset Delta', linewidth=1)
    ax1.plot(dataset_df['timestamp'], dataset_df['binance_delta'], alpha=0.7, label='Binance Delta', linewidth=1)
    # Mark extreme points
    extreme_df = dataset_df[(dataset_df['binance_delta'] < -50) | (dataset_df['binance_delta'] > 50)]
    if len(extreme_df) > 0:
        ax1.scatter(extreme_df['timestamp'], extreme_df['binance_delta'], color='red', s=50, alpha=0.8, label='Extreme Points', zorder=5)
    ax1.set_ylabel('Delta')
    ax1.set_title('Dataset Delta vs Binance Delta Over Time (Around Extreme Points)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Bottom plot: bid vs timestamp
    ax2.plot(dataset_df['timestamp'], dataset_df['bid'], alpha=0.7, label='Bid', linewidth=1, color='green')
    ax2.set_xlabel('Timestamp')
    ax2.set_ylabel('Bid')
    ax2.set_title('Bid Over Time')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show(block=True)
    
    return binance_df, dataset_df

def view_price_binance_and_dataset(binance_df, dataset_df):
    """Match Binance and dataset by timestamp and plot close price vs price"""
    binance_df = binance_df.sort_values('open_time').copy()
    print(f"binance_df: {len(binance_df)}")
    dataset_df = dataset_df.sort_values('timestamp').copy()
    print(f"dataset_df: {len(dataset_df)}")
    
    # Create a mapping from open_time to close price in binance_df
    binance_dict = dict(zip(binance_df['open_time'], binance_df['close']))
    print(f"binance_dict: {len(binance_dict)}")
    # Match dataset_df['timestamp'] with binance_df['open_time'] and add binance_close
    dataset_df['binance_close'] = dataset_df['timestamp'].map(binance_dict)

    
    # Remove rows where we couldn't find a match
    dataset_df = dataset_df.dropna(subset=['price', 'binance_close'])
    print(f"matched rows: {len(dataset_df)}")
    
    # Limit to first 300 points for plotting
    if len(dataset_df) > 300:
        dataset_df = dataset_df.head(300)
        print(f"limited to first 300 points for plotting")
    
    # Create subplots: price plot on top, bid plot on bottom
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    
    # Top plot: timestamp on x-axis with price and binance_close as two lines
    ax1.plot(dataset_df['timestamp'], dataset_df['price'], alpha=0.7, label='Dataset Price', linewidth=1)
    ax1.plot(dataset_df['timestamp'], dataset_df['binance_close'], alpha=0.7, label='Binance Close', linewidth=1)
    ax1.set_ylabel('Price')
    ax1.set_title('Dataset Price vs Binance Close Price Over Time')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Bottom plot: bid vs timestamp
    ax2.plot(dataset_df['timestamp'], dataset_df['bid'], alpha=0.7, label='Bid', linewidth=1, color='green')
    ax2.set_xlabel('Timestamp')
    ax2.set_ylabel('Bid')
    ax2.set_title('Bid Over Time')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show(block=True)
    
    return binance_df, dataset_df

    
  
 
def main():
    dataset = Dataset()
    df = dataset.df
    df['delta'] = df['price'] - df['price'].shift(1)
    binance = Binance(symbol="BTCUSDT", interval="1s")
    binance_df = binance.df
    binance_df['delta'] = binance_df['close'] - binance_df['close'].shift(1)

    view_delta_binance_and_dataset(binance_df, df)


    return 
if __name__ == "__main__":
    main()