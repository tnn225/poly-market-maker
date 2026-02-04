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

def show_pnl_by_interval(df):
    df = df[(df['delta'] >= 0) & (df['bid'] <= 0.49)]
    df['pnl'] = df['is_up'] - df['bid']
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
    print(f"Total PnL: {total_pnl} volume {volume} pnl/volume {total_pnl/volume:.2f} num_intervals {num_intervals} pnl/interval {total_pnl/num_intervals:.2f}")

def main():

    intervals = Interval()
    dataset = Dataset()
    df = dataset.df 

    show_pnl_by_interval(df)

if __name__ == "__main__":
    main()