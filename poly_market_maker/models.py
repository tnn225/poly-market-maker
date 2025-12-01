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
from poly_market_maker.tensorflow_classifier import TensorflowClassifier
from poly_market_maker.bucket_classifier import BucketClassifier

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
    def __init__(self, name, model, feature_cols, dataset=None):
        self.name = name
        self.model = model
        self.feature_cols = feature_cols
        self.dataset = dataset
        self.is_tensorflow = isinstance(self.model, TensorflowClassifier)
        self.filename = f'./data/models/{self.name}.keras' if self.is_tensorflow else f'./data/models/{self.name}.pkl'
        if os.path.exists(self.filename):
            print(f"Model file exists: {self.filename} {self.model}")
            self.model = self.load()
        else:
            self.fit(dataset.train_df[self.feature_cols], dataset.train_df['label'])

    def fit(self, X, y):
        # if hasattr(X, "values"):
        #     X = X.values
        # if hasattr(y, "values"):
        #    y = y.values
        self.model.fit(X, y)
        self.save()
        return self

    def predict_proba(self, X):
        #if hasattr(X, "values"):
        #    X = X.values
        return self.model.predict_proba(X)

    def evaluate(self, X, y):
        return self.model.evaluate(X, y)

    def save(self):
        # Check if this is a TensorFlow model (TensorflowClassifier wrapper)
        if isinstance(self.model, TensorflowClassifier):
            # Save the underlying TensorFlow model
            if self.model.model is not None:
                self.model.model.save(self.filename)
        else:
            pickle.dump(self.model, open(self.filename, 'wb'))

    def load(self):
        print(f"Loading model: {self.filename}")
        # Check if this is a TensorFlow model based on file extension or instance type
        if os.path.exists(self.filename):
            if self.is_tensorflow or self.filename.endswith('.keras'):
                # Load TensorFlow model into TensorflowClassifier wrapper
                if isinstance(self.model, TensorflowClassifier):
                    self.model.model = tf.keras.models.load_model(self.filename)
                else:
                    # Reconstruct TensorflowClassifier if needed
                    loaded_classifier = TensorflowClassifier()
                    loaded_classifier.model = tf.keras.models.load_model(self.filename)
                    self.model = loaded_classifier
            else:
                self.model = pickle.load(open(self.filename, 'rb'))
        else:
            raise ValueError(f"Model file not found: {self.filename}")
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

        # Create DataFrame instead of numpy array
        X = pd.DataFrame([row], columns=self.feature_cols)
        probability = self.model.predict_proba(X).flatten()[1]
        print(f"price: {price}, delta: {price - target}, target: {target}, seconds_left: {seconds_left}, bid: {bid}, ask: {ask}, probability: {probability}")

        # No trading
        # if row['delta'] < -100 or 100 < row['delta']:
        #     return row['bid']

        probability = float(np.clip(probability, 0.0, 1.0))
        if row['delta'] >= 0:
            probability = max(0.5, probability)
        if row['delta'] <= 0:
            probability = min(0.5, probability)
        return probability

def main():
    dataset = Dataset()
    train_df = dataset.train_df
    test_df = dataset.test_df

    # feature_cols=['delta', 'percent', 'log_return', 'time', 'seconds_left', 'bid', 'ask']
    feature_cols = ['delta', 'percent', 'log_return', 'time', 'seconds_left', 'bid', 'ask']

    models = []
 
    models.append(Model("BucketClassifier", BucketClassifier(), feature_cols=feature_cols, dataset=dataset))
    model = models[0]
    print(model.get_probability(87684.42122177457, 87498.58994751809, 60, 0.53, 0.55))
    print(model.get_probability(87398.58994751809, 87584.42122177457, 60, 0.45, 0.47))

if __name__ == "__main__":
    main()