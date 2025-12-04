from datetime import datetime, timezone, timedelta
import os
import time
import pandas as pd
import numpy as np
import logging
import sklearn.calibration
import csv

from collections import deque

from lightgbm import LGBMClassifier

from scipy.stats import norm
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False
    print("Warning: seaborn not available, using matplotlib for heatmap")
from sklearn.metrics import brier_score_loss, log_loss

from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)

from poly_market_maker.dataset import Dataset
from poly_market_maker.models import Model
from poly_market_maker.tensorflow_classifier import TensorflowClassifier
from poly_market_maker.bucket_classifier import BucketClassifier
from poly_market_maker.bid_classifier import BidClassifier

SPREAD = 0.05
SECONDS_LEFT_BIN_SIZE = 15
SECONDS_LEFT_BINS = int(900 / SECONDS_LEFT_BIN_SIZE)

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s"
)

FEATURE_COLS = ['delta', 'percent', 'log_return', 'time', 'seconds_left', 'bid', 'ask']

def show_delta_by_bid(df, SECONDS_LEFT):
    df = df[(df['seconds_left'] <= SECONDS_LEFT) & (df['seconds_left'] >= SECONDS_LEFT - 450.0)]
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    QUANTILES = [0.05, 0.50, 0.75, 0.9, 0.95]
    for quantile in QUANTILES:
        delta_by_bid = df.groupby('bid')['delta'].quantile(quantile)
        ax.scatter(delta_by_bid.index, delta_by_bid.values, 
                  label=f'Q{quantile:.2f}', alpha=0.7, s=50)
    ax.set_xlabel('Bid')
    ax.set_ylabel('Delta')
    ax.set_title(f'Delta Quantiles by Bid (seconds_left: {SECONDS_LEFT - 450.0:.0f} - {SECONDS_LEFT:.0f})')
    ax.legend(title='Quantiles', bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('delta_by_bid_all_quantiles.png', dpi=150, bbox_inches='tight')
    plt.show()

def print_outlier_by_bid(df, SECONDS_LEFT):
    df = df[(df['seconds_left'] <= SECONDS_LEFT) & (df['seconds_left'] >= SECONDS_LEFT - 450.0)]
    df = df[(df['delta'] > 100) & (df['bid'] < 0.5)]
    print(f"Outlier by bid: {df.head()}")
    for index, row in df.iterrows():
        print(f"Interval: {row['interval']}, Delta: {row['delta']}, Bid: {row['bid']}, Ask: {row['ask']}")

def get_bids_by_seconds_left_bin(df, quantile, seconds_left_bin):


    df = df[(df['seconds_left'] <= SECONDS_LEFT) & (df['seconds_left'] >= SECONDS_LEFT - 450.0)]
    
    delta_by_bid = df.groupby('bid')['delta'].quantile(quantile)
    for bid, delta in delta_by_bid.items():
        print(f"Bid: {bid:.2f} Delta: {delta:.2f}")
    print(f"Bids: {delta_by_bid.index} Quantile: {quantile:.2f} Delta: {delta_by_bid.values}")

def get_seconds_left_bin(seconds_left):
    return int(seconds_left // 15)

def get_bid_bin(bid):
    return int(bid * 100)


def init(df):
    df['seconds_left_bin'] = df['seconds_left'].apply(get_seconds_left_bin)
    return df

def get_delta_thresholds(df, quantile):
    thresholds = df.groupby(['seconds_left_bin', 'bid'])['delta'].quantile(quantile)
    plot_thresholds(thresholds)
    return thresholds

def plot_thresholds(thresholds):
    seconds_left_bins = [0, 30, 60] # i for i in range(0, SECONDS_LEFT_BINS, 10) ]
    for seconds_left_bin, bid in thresholds.index:
        if seconds_left_bin in seconds_left_bins:
            #print(f"Seconds left bin: {seconds_left_bin}, Bid: {bid}, Delta: {thresholds[seconds_left_bin][bid]}")
            plt.scatter(bid, thresholds[seconds_left_bin][bid])
    plt.show()

def evaluate(df, quantile):
    thresholds = get_delta_thresholds(df, quantile)

    df['pnl'] = df['is_up'] - df['bid']

    df['action'] = 0
    for index, row in df.iterrows():
        interval = row['interval']
        seconds_left_bin = row['seconds_left_bin']
        bid = row['bid']
        delta = row['delta']
        if bid in thresholds[seconds_left_bin] and delta >= thresholds[seconds_left_bin][bid]:
            df.at[index, 'action'] = 1
            # print(f"Interval: {interval}, seconds_left_bin: {seconds_left_bin}, Bid: {bid}, Delta: {delta}, Action: 1")

    df = df[df['action'] == 1]
    grouped = df.groupby('interval')['pnl'].agg(['sum', 'count'])

    total_pnl = 0
    volume = 0
    num_intervals = 0
    for interval, row in grouped.iterrows():
        pnl = row['sum']
        count = row['count']
        print(f"Interval: {interval}, PnL: {pnl}, Count: {count}")
        total_pnl += pnl
        volume += count
        num_intervals += 1
    print(f"Quantile: {quantile:.2f} PnL: {total_pnl} {total_pnl/volume:.2f} volume {volume} num_intervals {num_intervals}")

def main():
    dataset = Dataset()
    train_df = init(dataset.train_df)
    test_df = init(dataset.test_df)

    # for quantile in [0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.9, 0.95, 0.99]:
    for quantile in [0.99]:
        evaluate(test_df, quantile)
    
if __name__ == "__main__":
    main()

