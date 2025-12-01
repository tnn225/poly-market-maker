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
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.base import clone
import optuna

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

def run_grid_search(dataset):
    feature_cols = ['delta', 'percent', 'log_return', 'time', 'seconds_left', 'bid', 'ask']
    X_train = dataset.train_df[feature_cols]
    y_train = dataset.train_df['label']

    param_grid = {
        'n_estimators': [100, 300, 500, 700, 900, 1000],
        'max_depth': [5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, None],
        'min_samples_split': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000],
        'min_samples_leaf': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000],
    }

    base_rf = RandomForestClassifier(random_state=42, n_jobs=-1)
    grid_search = GridSearchCV(
        estimator=base_rf,
        param_grid=param_grid,
        cv=3,
        scoring='roc_auc',
        n_jobs=-1,
        verbose=2,
        refit=True,
    )

    logger.info("Starting GridSearchCV for RandomForestClassifier...")
    grid_search.fit(X_train, y_train)
    logger.info(f"Best ROC-AUC: {grid_search.best_score_:.4f}")
    logger.info(f"Best Params: {grid_search.best_params_}")

    best_params = grid_search.best_params_
    best_rf = RandomForestClassifier(
        **{k: best_params[k] for k in ['n_estimators', 'max_depth', 'min_samples_split', 'min_samples_leaf']},
        random_state=42,
        n_jobs=-1,
    )
    best_model = Model(
        "RandomForestClassifier_Best",
        best_rf,
        feature_cols=feature_cols,
        dataset=dataset,
    )
    evaluate_model(best_model, dataset.test_df.copy(), show_results=True)


def run_optuna_search(dataset, n_trials=10):
    feature_cols = ['delta', 'percent', 'log_return', 'time', 'seconds_left', 'bid', 'ask']
    X_train = dataset.train_df[feature_cols]
    y_train = dataset.train_df['label']

    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 1000, 2000),
            "max_depth": trial.suggest_int("max_depth", 30, 100),
            "min_samples_split": trial.suggest_int("min_samples_split", 50, 200),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 50, 200),
            "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", 0.5, 1.0]),
            "bootstrap": trial.suggest_categorical("bootstrap", [False]),
            "class_weight": trial.suggest_categorical("class_weight", ["balanced", "balanced_subsample"]),
            "random_state": 42,
            "n_jobs": -1,
        }
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        pnls = []
        for train_idx, val_idx in skf.split(X_train, y_train):
            X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
            y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
            model = RandomForestClassifier(**params)
            model.fit(X_tr, y_tr)
            probs = model.predict_proba(X_val)[:, 1]
            val_df = X_val.copy()
            val_df = val_df.assign(probability=probs, is_up=y_val.values.astype(float))
            action = (val_df['probability'] - SPREAD > val_df['bid']) & (val_df['bid'] > 0)
            pnl = float((val_df.loc[action, 'is_up'] - val_df.loc[action, 'bid']).sum()) if action.any() else 0.0
            pnls.append(pnl)
        return float(np.mean(pnls))

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    logger.info(f"Optuna best PnL: {study.best_value:.4f}")
    logger.info(f"Optuna best params: {study.best_params}")

    best_params = study.best_params
    best_rf = RandomForestClassifier(**best_params)
    best_model = Model(
        "RandomForestClassifier_Optuna",
        best_rf,
        feature_cols=feature_cols,
        dataset=dataset,
    )
    evaluate_model(best_model, dataset.test_df.copy(), show_results=True)


def main():
    dataset = Dataset()
    # run_grid_search(dataset)
    run_optuna_search(dataset)


if __name__ == "__main__":
    main()