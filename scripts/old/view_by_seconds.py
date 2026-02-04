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

def main():
    SECONDS_LEFT = 450
    dataset = Dataset()
    show_delta_by_bid(dataset.df, SECONDS_LEFT)
    print_outlier_by_bid(dataset.df, SECONDS_LEFT)
    
if __name__ == "__main__":
    main()