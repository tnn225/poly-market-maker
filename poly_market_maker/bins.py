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
    return int(delta / 1)

def get_seconds_left_bin(seconds_left):
    return int(seconds_left / 15)

def init_bins(df):
    # Filter delta to be between -100 and 100
    df = df[(df['delta'] >= -50) & (df['delta'] <= 50)].copy()
    
    df['delta_bin'] = df['delta'].apply(get_delta_bin)
    df['seconds_left_bin'] = df['seconds_left'].apply(get_seconds_left_bin)
    df['pnl'] = df['is_up'] - df['bid']
    return df


def plot_delta_distribution(df):
    
    df.groupby(['delta_bin', 'seconds_left_bin']).size().unstack().plot(kind='bar', stacked=True)
    plt.xlabel('Delta Bin')
    plt.ylabel('Frequency')
    plt.title('Delta Distribution by Seconds Left')
    plt.show()
    return df

def show_pnl(df):
    df = init_bins(df)
    print(df.head())

   # Calculate PnL for each group
    group_results = []
    for (t_bucket, d_bucket), group in df.groupby(['seconds_left_bin', 'delta_bin'], observed=False):
        if d_bucket != -10:
            continue

        group_results.append({
            'seconds_left_bucket': t_bucket,
            'delta_bucket': d_bucket,
            'pnl': round(group['is_up'].sum() - group['bid'].sum(), 3),
            'average_revenue': round(group['is_up'].mean(), 3),
            'average_cost': round(group['bid'].mean(), 3),
            'margin': round(group['pnl'].mean(), 3) * 100,

            'num_rows': len(group),
            'num_intervals': group['interval'].nunique(),
            # Count unique intervals where is_up is True
            'up / intervals': round(group.loc[group['is_up'] == True, 'interval'].nunique() / group['interval'].nunique(), 3),
        })
    
    # Create results dataframe and sort by increasing PnL
    results_df = pd.DataFrame(group_results)
    # results_df = results_df.sort_values(['pnl', 'seconds_left_bucket', 'delta_bucket'], ascending=[True, True, True])
    results_df = results_df.sort_values(['seconds_left_bucket', 'delta_bucket'], ascending=[False, True])
    
    print("\nGroups sorted by seconds_left_bucket and average_revenue:")
    print(results_df.to_string(index=False))

if __name__ == "__main__":
    dataset = Dataset()
    train_df = dataset.train_df
    test_df = dataset.test_df

    # plot_delta_distribution(train_df)
    # plot_delta_distribution(test_df)
    # plt.show()

    # show_pnl(train_df)
    show_pnl(test_df)