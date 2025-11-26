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

SPREAD = 0.1
DAYS = 7
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s"
)


def main():
    dataset = Dataset()
    
    train_df = dataset.train_df
    test_df = dataset.test_df

    model = Model("RandomForestClassifier", RandomForestClassifier(n_estimators=1000, max_depth=10, min_samples_split=100, random_state=42, n_jobs=-1))
    feature_cols = model.feature_cols
    probs = model.predict_proba(test_df[feature_cols])
    test_df['probability'] = np.clip(probs, 0.0, 1.0)

    y_true = test_df['label'].astype(int)
    y_pred_proba = test_df['probability'].astype(float)

    df = test_df.copy()
    df['y_true'] = y_true
    df['y_pred_proba'] = y_pred_proba

    # Bucket price_delta into 10 bins
    df['delta_bucket'] = pd.qcut(test_df['delta'], 20, duplicates='drop')
    
    # Bucket seconds_left into 5 bins
    df['time_bucket'] = pd.qcut(test_df['time'], 5, duplicates='drop')

    # Calculate PnL for each group
    group_results = []
    for (d_bucket, t_bucket), group in df.groupby(['delta_bucket', 'time_bucket']):
        action = (group['probability'] - SPREAD > group['bid']) & (group['bid'] > 0)
        buy_trades = group[action].copy()
        
        buy_trades['revenue'] = buy_trades['is_up'].astype(float)
        buy_trades['cost'] = buy_trades['bid'].astype(float)
        buy_trades['pnl'] = buy_trades['revenue'] - buy_trades['cost']
        
        group_results.append({
            'delta_bucket': d_bucket,
            'time_bucket': t_bucket,
            'revenue': group['is_up'].sum(),
            'margin': (group['is_up'].sum() - group['bid'].sum()) / group['is_up'].sum() * 100,
            'average_probability': group['probability'].mean(),
            'average_revenue': group['is_up'].mean(),
            'average_cost': group['bid'].mean(),
            'total_revenue': buy_trades['revenue'].sum(),
            'total_cost': buy_trades['cost'].sum(),
            'total_pnl': buy_trades['pnl'].sum(),
            'buy_trades': len(buy_trades),
            'num_rows': len(group),
        })
    
    # Create results dataframe and sort by increasing PnL
    results_df = pd.DataFrame(group_results)
    results_df = results_df.sort_values('average_probability', ascending=True)
    
    print("\nGroups sorted by increasing PnL:")
    print(results_df.to_string(index=False))

    print(buy_trades.head())


if __name__ == "__main__":
    main()