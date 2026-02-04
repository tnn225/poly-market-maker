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
from poly_market_maker.models.tensorflow_classifier import TensorflowClassifier
from poly_market_maker.models.bucket_classifier import BucketClassifier
from poly_market_maker.models.bid_classifier import BidClassifier

SPREAD = 0.05
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s"
)

FEATURE_COLS = ['delta', 'percent', 'log_return', 'time', 'seconds_left', 'bid', 'ask']

def show_delta(df):
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    ax.hist(df['delta'].dropna(), bins=200, alpha=0.7, edgecolor='black')
    ax.set_xlabel('delta', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title('Distribution of Delta', fontsize=14)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def show_interval_min_max(df):
    # Group by interval to get min and max delta, and get the timestamp for each interval
    grouped = df.groupby('interval').agg({
        'delta': ['min', 'max'],
        'timestamp': 'first'  # Get the first timestamp for each interval
    })
    
    # Flatten column names
    grouped.columns = ['delta_min', 'delta_max', 'timestamp']
    grouped = grouped.reset_index()
    grouped = grouped.sort_values('timestamp')
    
    # Print statistics about intervals with min_delta = 0 and max_delta = 0
    total_intervals = len(grouped)
    intervals_min_zero = len(grouped[grouped['delta_min'] >= 0])
    intervals_max_zero = len(grouped[grouped['delta_max'] <= 0])
    
    print(f"Number of intervals with min_delta = 0: {intervals_min_zero} ({intervals_min_zero/total_intervals*100:.2f}%)")
    print(f"Number of intervals with max_delta = 0: {intervals_max_zero} ({intervals_max_zero/total_intervals*100:.2f}%)")
    
    # Plot min and max lines
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    ax.plot(grouped['timestamp'], grouped['delta_min'], label='Min Delta', linewidth=1, alpha=0.7)
    ax.plot(grouped['timestamp'], grouped['delta_max'], label='Max Delta', linewidth=1, alpha=0.7)
    ax.fill_between(grouped['timestamp'], grouped['delta_min'], grouped['delta_max'], alpha=0.2)
    
    ax.set_xlabel('Timestamp', fontsize=12)
    ax.set_ylabel('Delta', fontsize=12)
    ax.set_title('Min and Max Delta by Interval', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def main():
    dataset = Dataset()
    train_df = dataset.train_df
    test_df = dataset.test_df

    # Plot min and max delta by interval
    if 'interval' in train_df.columns and 'delta' in train_df.columns:
        show_interval_min_max(train_df)
  
if __name__ == "__main__":
    main()
