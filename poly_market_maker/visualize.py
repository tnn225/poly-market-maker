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

def main():
    dataset = Dataset()
    train_df = dataset.train_df
    test_df = dataset.test_df

    # Plot delta distribution
    if 'delta' in train_df.columns:
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        
        # Plot histogram of delta values
        ax.hist(train_df['log_return'].dropna(), bins=200, alpha=0.7, edgecolor='black')
        ax.set_xlabel('log_return', fontsize=12)
        ax.set_ylabel('Frequency', fontsize=12)
        ax.set_title('Distribution of Delta in Training Data', fontsize=14)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        # Print some statistics
        print(f"Delta Statistics:")
        print(train_df['delta'].describe())
    else:
        print("'delta' column not found in train_df")
    

if __name__ == "__main__":
    main()
