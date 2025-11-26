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


import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from poly_market_maker.dataset import Dataset

SPREAD = 0.1
DAYS = 7
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s"
)


def main():
    dataset = Dataset()
    feature_cols = dataset.feature_cols
    train_df = dataset.train_df
    test_df = dataset.test_df

    model = RandomForestClassifier(n_estimators=300, max_depth=10, min_samples_split=50, random_state=42, n_jobs=-1)
    model.fit(train_df[feature_cols], train_df['label'])
    probs = model.predict_proba(test_df[feature_cols])[:, 1]
    test_df['probability'] = probs

    y_true = test_df['label'].astype(int)
    y_pred_proba = test_df['probability'].astype(float)

    prob_true, prob_pred = sklearn.calibration.calibration_curve(y_true, y_pred_proba, n_bins=10)

    plt.figure(figsize=(6, 6))
    plt.plot(prob_pred, prob_true, marker='o', linewidth=1, label='Model')
    plt.plot([0, 1], [0, 1], '--', label='Perfect Calibration')
    plt.xlabel('Predicted Probability')
    plt.ylabel('Actual Frequency')
    plt.title('Calibration Curve')
    plt.legend()
    plt.show()

    print("Brier Score:", brier_score_loss(y_true, y_pred_proba))

    df = test_df.copy()
    df['y_true'] = y_true
    df['y_pred_proba'] = y_pred_proba

    # Bucket price_delta into 10 bins
    df['delta_bucket'] = pd.qcut(test_df['delta'], 3, duplicates='drop')
    
    # Bucket seconds_left into 5 bins
    df['time_bucket'] = pd.qcut(test_df['time'], 3, duplicates='drop')

    # Calculate PnL for each group
    group_results = []
    for (d_bucket, t_bucket), group in df.groupby(['delta_bucket', 'time_bucket']):
        action = (group['probability'] - SPREAD > group['bid']) & (group['bid'] > 0)
        buy_trades = group[action].copy()
        
        if len(buy_trades) > 0:
            buy_trades['revenue'] = buy_trades['is_up'].astype(float)
            buy_trades['cost'] = buy_trades['bid'].astype(float)
            buy_trades['pnl'] = buy_trades['revenue'] - buy_trades['cost']
            total_pnl = buy_trades['pnl'].sum()
        else:
            total_pnl = 0.0
        
        group_results.append({
            'delta_bucket': d_bucket,
            'time_bucket': t_bucket,
            'total_pnl': total_pnl,
            'num_trades': len(buy_trades) if len(buy_trades) > 0 else 0,
            'num_rows': len(group)
        })
    
    # Create results dataframe and sort by increasing PnL
    results_df = pd.DataFrame(group_results)
    results_df = results_df.sort_values('total_pnl', ascending=True)
    
    print("\nGroups sorted by increasing PnL:")
    print(results_df.to_string(index=False))




if __name__ == "__main__":
    main()