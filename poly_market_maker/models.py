from datetime import datetime, timezone, timedelta
import os
import time
import pandas as pd
import numpy as np
import logging

import csv

from collections import deque

from lightgbm import LGBMClassifier

from scipy.stats import norm
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

from poly_market_maker.dataset import Dataset

import pickle

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

SPREAD = 0.01
DAYS = 7
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s"
)


class Model:
    def __init__(self, name, model, feature_cols, dataset=None):
        self.name = name
        self.model = model
        self.feature_cols = feature_cols
        self.dataset = dataset
        self.filename = f'./data/models/{self.name}.pkl'
        if os.path.exists(self.filename):
            self.model = self.load()
        else:
            self.fit(dataset.train_df[self.feature_cols], dataset.train_df['label'])

    def fit(self, X, y):
        if hasattr(X, "values"):
            X = X.values
        if hasattr(y, "values"):
            y = y.values
        self.model.fit(X, y)
        self.save()
        return self

    def predict_proba(self, X):
        if hasattr(X, "values"):
            X = X.values
        return self.model.predict_proba(X)

    def evaluate(self, X, y):
        return self.model.evaluate(X, y)

    def save(self):
        pickle.dump(self.model, open(self.filename, 'wb'))

    def load(self):
        if os.path.exists(self.filename):
            self.model = pickle.load(open(self.filename, 'rb'))
        return self.model

    def get_probability(self, price, target, seconds_left, bid, ask):
        """Generate a single probability prediction for the provided snapshot."""
        if self.model is None:
            raise ValueError("Model has not been trained yet. Call train() first.")

        if target <= 0 or price <= 0:
            raise ValueError("price and target must be positive numbers.")

        row = {
            'delta': float(price - target),
            'percent': float(price - target) / target,
            'log_return': float(np.log(price / target)),
            'time': float(seconds_left / 900.0),
            'seconds_left': float(seconds_left),
            'bid': float(bid),
            'ask': float(ask),
        }
   
        X = np.array([[row[k] for k in self.feature_cols]], dtype='float32')
        probability = self.model.predict_proba(X).flatten()[1]
        print(f"price: {price}, delta: {price - target}, target: {target}, seconds_left: {seconds_left}, bid: {bid}, ask: {ask}, probability: {probability}")
        return float(np.clip(probability, 0.0, 1.0))

def main():
    dataset = Dataset()
    train_df = dataset.train_df
    test_df = dataset.test_df

    N_ESTIMATORS = 200
    MAX_DEPTH = 6
    models = [
        Model("RandomForestClassifier", RandomForestClassifier(n_estimators=N_ESTIMATORS, max_depth=MAX_DEPTH, min_samples_split=100, random_state=42, n_jobs=-1)),
        # Model("LogisticRegression", LogisticRegression(penalty='l2', C=1.0, solver='lbfgs', max_iter=10000)),
        # Model("LGBMClassifier", LGBMClassifier(n_estimators=N_ESTIMATORS, max_depth=-1, learning_rate=0.001, random_state=42,)),
        # Model("GradientBoostingClassifier", GradientBoostingClassifier(n_estimators=N_ESTIMATORS, learning_rate=0.01, max_depth=MAX_DEPTH, random_state=42)),
    ]

    for model in models:
        feature_cols = model.feature_cols

        if os.path.exists(model.filename):
            model.model = model.load()
        else:
            model.fit(train_df[feature_cols], train_df['label'])

        if hasattr(model.model, "predict_proba"):
            probs = model.model.predict_proba(test_df[feature_cols])[:, 1]
        else:
            probs = model.model.predict(test_df[feature_cols])
            probs = np.clip(probs, 0.0, 1.0)
        test_df[f'prob_{model.name}'] = probs
        metrics = dataset.evaluate_model_metrics(test_df, probability_column=f'prob_{model.name}', spread=SPREAD)
        metrics['model'] = model.name
        print(datetime.now(), metrics)

    model = models[0]
    print(model.get_probability(87684.42122177457, 87498.58994751809, 60, 0.53, 0.55))
    print(model.get_probability(87398.58994751809, 87584.42122177457, 60, 0.45, 0.47))

if __name__ == "__main__":
    main()