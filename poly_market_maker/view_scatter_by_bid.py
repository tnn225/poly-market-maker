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

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s"
)

FEATURE_COLS = ['delta', 'percent', 'log_return', 'time', 'seconds_left', 'bid', 'ask']

def get_delta_bin(delta):
    """Convert delta to bin number"""
    return int(delta / 10)

def show_scatter_plot(df, bid_threshold=0.01):
    """Create scatter plot: x-axis seconds_left_bin, y-axis delta_bin
    Green points for is_up=True, red points for is_up=False"""

    # df = df[((df['bid'] == 0.01) & (df['label'] == 1))]
    df = df[(df['bid'] == bid_threshold)]
    
    # Create delta_bin if it doesn't exist
    if 'delta_bin' not in df.columns:
        df = df.copy()
        df['delta_bin'] = df['delta'].apply(get_delta_bin)
    
    # Ensure seconds_left_bin exists
    if 'seconds_left_bin' not in df.columns:
        from poly_market_maker.dataset import SECONDS_LEFT_BIN_SIZE
        df = df.copy()
        df['seconds_left_bin'] = df['seconds_left'].apply(lambda x: int(x // SECONDS_LEFT_BIN_SIZE))
    
    # Filter out rows with missing values
    df = df.dropna(subset=['seconds_left_bin', 'delta_bin', 'is_up'])
    
    # Separate data by is_up
    df_up = df[df['is_up'] == True]
    df_down = df[df['is_up'] == False]
    
    fig, ax = plt.subplots(figsize=(12, 8))

    if bid_threshold < 0.5:
        ax.scatter(df_down['seconds_left_bin'], df_down['delta_bin'], 
                  c='red', alpha=0.5, s=10, label='DOWN')
        ax.scatter(df_up['seconds_left_bin'], df_up['delta_bin'], 
                  c='green', alpha=0.5, s=10, label='UP')
    else:
        ax.scatter(df_up['seconds_left_bin'], df_up['delta_bin'], 
                  c='green', alpha=0.5, s=10, label='UP')
        ax.scatter(df_down['seconds_left_bin'], df_down['delta_bin'], 
                  c='red', alpha=0.5, s=10, label='DOWN')
    
    ax.set_xlabel('seconds_left_bin')
    ax.set_ylabel('delta_bin')
    ax.set_title(f'Scatter Plot bid = {bid_threshold}: seconds_left_bin vs delta_bin')
    ax.invert_xaxis()  # Flip x-axis so seconds_left_bin is decreasing
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'./data/plots/scatter_seconds_left_bin_delta_bin_bid_{bid_threshold}.png', dpi=150, bbox_inches='tight')
    plt.show()

def main():
    dataset = Dataset()
    show_scatter_plot(dataset.df, bid_threshold=0.25)

if __name__ == "__main__":
    main()

