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

def run_optuna_search(dataset, n_trials=10):
    feature_cols = ['delta', 'percent', 'log_return', 'time', 'seconds_left', 'bid', 'ask']
    X_train = dataset.train_df[feature_cols]
    y_train = dataset.train_df['label']

    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 200),
            "max_depth": trial.suggest_int("max_depth", 30, 100),
            "min_samples_split": trial.suggest_int("min_samples_split", 50, 200),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 50, 200),
            "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", 0.5, 1.0]),
            "bootstrap": trial.suggest_categorical("bootstrap", [False]),
            "class_weight": trial.suggest_categorical("class_weight", ["balanced", "balanced_subsample"]),
            "random_state": 42,
            "n_jobs": -1,
        }
        print(f"Trial {trial.number} params: {params}")
        skf = StratifiedKFold(n_splits=5, shuffle=False)
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

    from poly_market_maker import evaluate
    evaluate.evaluate_model(best_model, dataset, show_results=True)

def run_optuna_search_lgbm(dataset, n_trials=100):
    """Optuna hyperparameter search for LGBMClassifier optimized for PnL"""
    feature_cols = ['delta', 'percent', 'log_return', 'time', 'seconds_left', 'bid', 'ask']
    X_train = dataset.train_df[feature_cols]
    y_train = dataset.train_df['label']

    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 2000),
            "max_depth": trial.suggest_int("max_depth", 3, 15),
            "learning_rate": trial.suggest_float("learning_rate", 0.001, 0.1, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 16, 256),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "min_child_samples": trial.suggest_int("min_child_samples", 10, 200),
            "reg_alpha": trial.suggest_float("reg_alpha", 0.0001, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.0001, 10.0, log=True),
            "random_state": 42,
            "n_jobs": -1,
            "verbosity": -1,
        }
        print(f"Trial {trial.number} params: {params}")
        skf = StratifiedKFold(n_splits=5, shuffle=False)
        pnls = []
        for train_idx, val_idx in skf.split(X_train, y_train):
            X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
            y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
            model = LGBMClassifier(**params)
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
    best_lgbm = LGBMClassifier(**best_params)
    best_model = Model(
        "LGBMClassifier_Optuna",
        best_lgbm,
        feature_cols=feature_cols,
        dataset=dataset,
    )

    from poly_market_maker import evaluate
    evaluate.evaluate_model(best_model, dataset, show_results=True)
    
    return best_params, best_model

def main():
    dataset = Dataset()
    # run_grid_search(dataset)
    # run_optuna_search(dataset)
    run_optuna_search_lgbm(dataset, n_trials=100)


if __name__ == "__main__":
    main()