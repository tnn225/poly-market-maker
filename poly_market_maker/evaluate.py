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

SPREAD = 0.05
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s"
)

def evaluate_model(model, test_df, show_results=True):
    feature_cols = model.feature_cols
    probs = model.predict_proba(test_df[feature_cols])[:, 1]
    test_df['probability'] = np.clip(probs, 0.0, 1.0)
    y_true = test_df['label'].astype(float)
    y_pred_proba = test_df['probability'].astype(float)
    test_df['y_true'] = y_true
    test_df['y_pred_proba'] = y_pred_proba

    # Bucket price_delta into 10 bins
    test_df['delta_bucket'] = pd.qcut(test_df['delta'], 5, duplicates='drop')
    
    # Bucket seconds_left into 5 bins
    test_df['time_bucket'] = pd.qcut(test_df['time'], 5, duplicates='drop')

    # Calculate PnL for each group
    group_results = []
    for (d_bucket, t_bucket), group in test_df.groupby(['delta_bucket', 'time_bucket'], observed=False):
        action = (group['probability'] - SPREAD > group['bid']) & (group['bid'] > 0)
        buy_trades = group[action].copy()
        
        buy_trades['revenue'] = buy_trades['is_up'].astype(float)
        buy_trades['cost'] = buy_trades['bid'].astype(float)
        buy_trades['pnl'] = buy_trades['revenue'] - buy_trades['cost']
        
        group_results.append({
            'time_bucket': t_bucket,
            'delta_bucket': d_bucket,
            

            'average_is_up': group['is_up'].mean(),
            'average_probability': group['probability'].mean(),
            'average_bid': group['bid'].mean(),
            'pnl': buy_trades['pnl'].sum(),  
            'buy_trades': len(buy_trades),
            'num_rows': len(group),
        })
    
    # Create results dataframe and sort by increasing PnL
    results_df = pd.DataFrame(group_results)
    results_df = results_df.sort_values(['time_bucket', 'delta_bucket'], ascending=[False, True])
    
    if show_results:
        print("\nGroups sorted by time bucket and delta bucket:")
        print(results_df.to_string(index=False))

    ret = model.dataset.evaluate_model_metrics(test_df, probability_column='probability', threshold=0.5, spread=SPREAD)
    ret['model'] = model.name
    print(ret)
    return ret

def main():
    dataset = Dataset()
    
    train_df = dataset.train_df
    test_df = dataset.test_df

    feature_cols = ['delta', 'time']
    models = [
        Model("RandomForestClassifier", RandomForestClassifier(n_estimators=300, max_depth=10, min_samples_split=100, random_state=42, n_jobs=-1), feature_cols, dataset=dataset),
        Model("LogisticRegression", LogisticRegression(max_iter=300, C=0.5, solver='lbfgs'), feature_cols, dataset=dataset),
        Model("LGBMClassifier", LGBMClassifier(n_estimators=300, max_depth=-1, learning_rate=0.01, random_state=42,), feature_cols, dataset=dataset),
        Model("GradientBoostingClassifier", GradientBoostingClassifier(n_estimators=300, learning_rate=0.01, max_depth=3, random_state=42), feature_cols, dataset=dataset),
        Model("TensorflowClassifier", TensorflowClassifier(filename='./data/models/tensorflow_classifier.keras'), feature_cols, dataset=dataset),
    ]

    for model in models:
        evaluate_model(model, test_df, show_results=False)



if __name__ == "__main__":
    main()