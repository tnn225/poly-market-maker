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
from sklearn.calibration import CalibratedClassifierCV

from poly_market_maker.dataset import Dataset
from poly_market_maker.tensorflow_classifier import TensorflowClassifier
from poly_market_maker.bucket_classifier import BucketClassifier
from poly_market_maker.delta_classifier import DeltaClassifier

import pickle
import tensorflow as tf

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
    def __init__(self, name, model=None, feature_cols=None, dataset=None):
        print(f"Initializing model: {name}")
        self.name = name
        self.model = model
        self.feature_cols = feature_cols
        self.filename = f'./data/models/{self.name}.pkl'
        if os.path.exists(self.filename):
            self.load()
        else:
            self.fit(dataset.train_df[feature_cols], dataset.train_df['label'])

    def fit(self, X, y):
        print(f"Fitting model: {self.name}")
        # if hasattr(X, "values"):
        #     X = X.values
        # if hasattr(y, "values"):
        #    y = y.values
        self.model.fit(X, y)
        self.save()
        return self

    def predict_proba(self, X):
        # if hasattr(X, "values"):
        #     X = X.values
        return self.model.predict_proba(X)

    def evaluate(self, X, y):
        return self.model.evaluate(X, y)

    def save(self):
        """Save both model and feature_cols"""
        print(f"Model saved to: {self.filename}")        
        # Save as a dictionary containing both model and feature_cols
        data_to_save = {
            'model': self.model,
            'feature_cols': self.feature_cols
        }
        pickle.dump(data_to_save, open(self.filename, 'wb'))

    def load(self):
        """Load both model and feature_cols"""
        print(f"Loading model: {self.filename}")
        if os.path.exists(self.filename):
            loaded_data = pickle.load(open(self.filename, 'rb'))
            self.model = loaded_data.get('model')
            self.feature_cols = loaded_data.get('feature_cols')

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

        # Create DataFrame instead of numpy array
        X = pd.DataFrame([row], columns=self.feature_cols)
        probability = self.model.predict_proba(X).flatten()[1]
        print(f"price: {price}, delta: {price - target}, target: {target}, seconds_left: {seconds_left}, bid: {bid}, ask: {ask}, probability: {probability}")

        # No trading
        # if row['delta'] < -100 or 100 < row['delta']:
        #     return row['bid']

        probability = float(np.clip(probability, 0.0, 1.0))
        if row['delta'] > 0 and probability < 0.5:
            probability = bid 
        if row['delta'] < 0 and probability > 0.5:
            probability = bid
        return probability

def get_calibrated_model(random_forest_params):
    model = RandomForestClassifier(**random_forest_params)
    calibrated_model = CalibratedClassifierCV(model, method='isotonic', cv=5)
    return calibrated_model

def main():
    dataset = Dataset()
    train_df = dataset.train_df
    test_df = dataset.test_df
    feature_cols=['delta', 'seconds_left', 'bid']

    random_forest_params = {'n_estimators': 2000, 'max_depth': 30, 'min_samples_split': 100, 'min_samples_leaf': 100, 'max_features': 'sqrt', 'bootstrap': False, 'class_weight': 'balanced'}
    random_forest_params = {'n_estimators': 1567, 'max_depth': 10, 'min_samples_split': 143, 'min_samples_leaf': 12, 'max_features': 'sqrt', 'bootstrap': False, 'class_weight': 'balanced'}
    model = Model(f"RandomForestClassifier_features_{len(feature_cols)}_{random_forest_params['n_estimators']}_{random_forest_params['max_depth']}_{random_forest_params['min_samples_split']}_{random_forest_params['min_samples_leaf']}_{random_forest_params['max_features']}_{random_forest_params['bootstrap']}_{random_forest_params['class_weight']}", RandomForestClassifier(**random_forest_params), feature_cols=feature_cols, dataset=dataset)
    # model = Model(f"CalibratedClassifierCV_features_{len(feature_cols)}_{random_forest_params['n_estimators']}_{random_forest_params['max_depth']}_{random_forest_params['min_samples_split']}_{random_forest_params['min_samples_leaf']}_{random_forest_params['max_features']}_{random_forest_params['bootstrap']}_{random_forest_params['class_weight']}", get_calibrated_model(random_forest_params), feature_cols=feature_cols, dataset=dataset)
    # model = Model(f"DeltaClassifier", DeltaClassifier(), feature_cols=feature_cols, dataset=dataset)

    prob = model.predict_proba(test_df[feature_cols])
    test_df['probability'] = prob[:, 1]
    ret = dataset.evaluate_model_metrics(test_df, probability_column='probability', spread=0.05)
    print(ret)

    print(model.get_probability(87684.42122177457, 87498.58994751809, 60, 0.53, 0.55))
    print(model.get_probability(87398.58994751809, 87584.42122177457, 60, 0.45, 0.47))

if __name__ == "__main__":
    main()

