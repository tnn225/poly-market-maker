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

from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    confusion_matrix,
    roc_auc_score,
)


import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

SPREAD = 0.01
DAYS = 30

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s"
)

class Dataset:
    def __init__(self):
        self._delta_percentiles = None
        self._read_dates()
        self._add_target_and_is_up()
        self._train_test_split()

        self.show()

    def show(self):
        print(self.train_df.head())
        print(self.test_df.head())

        print("Train/Test set overview:")
        print(f"  train_df shape: {self.train_df.shape}")
        print(f"  test_df shape:  {self.test_df.shape}")
        print(f"  train labels distribution:\n {self.train_df['label'].value_counts(normalize=True)} {self.train_df['label'].value_counts(normalize=False)}")
        print(f"  test labels distribution:\n {self.test_df['label'].value_counts(normalize=True)} {self.test_df['label'].value_counts(normalize=False)}\n")

    def _read_dates(self):
        today = datetime.now()
        dates = [(today - timedelta(days=i)).strftime("%Y-%m-%d") for i in range(DAYS)]
        # Load data from CSV files for each date
        dataframes = []
        for date in dates:
            df = self._read_rows(date)
            if df is not None and not df.empty:
                dataframes.append(df)

        # Combine all dataframes
        if dataframes:
            self.df = pd.concat(dataframes, ignore_index=True)
        else:
            self.df = pd.DataFrame(columns=["timestamp", "price", "bid", "ask"])

        self.df['timestamp'] = self.df['timestamp'].astype(int)
        self.df['price'] = self.df['price'].astype(float)
        self.df['bid'] = self.df['bid'].astype(float)
        self.df['ask'] = self.df['ask'].astype(float)


    def _read_rows(self, date):
        path = f"./data/price_{date}.csv"
        if not os.path.exists(path):
            logger.warning(f"File not found: {path}")
            return None
        
        try:
            df = pd.read_csv(path)
            return df
        except Exception as e:
            logger.error(f"Error reading {path}: {e}")
            return None

    def _add_target_and_is_up(self):
        # Initialize columns
        self.df['target'] = None
        self.df['is_up'] = None
        
        # Calculate 15-minute interval start time (rounded down to nearest 900 seconds)
        self.df['interval'] = self.df['timestamp'] // 900 * 900
        
        # Create price lookup dictionary for O(1) access
        price_dict = dict(zip(self.df['timestamp'], self.df['price']))
        
        # Calculate target and is_up for each unique interval
        interval_values = {}
        for interval in self.df['interval'].unique():
            target = price_dict.get(interval)
            
            if target is not None:
                # Get price at the end of the interval (interval + 900 seconds)
                next_interval_price = price_dict.get(interval + 900)
                
                if next_interval_price is not None:
                    # is_up = True if price goes up or stays same
                    interval_values[interval] = {
                        'target': target,
                        'is_up': target <= next_interval_price
                    }
        
        # Apply the same target and is_up to all rows in [interval, interval+900)
        for interval, values in interval_values.items():
            mask = (self.df['interval'] == interval)
            self.df.loc[mask, 'target'] = values['target']
            self.df.loc[mask, 'is_up'] = values['is_up']

        # Filter out rows where target or is_up couldn't be calculated
        self.df = self.df[
            (self.df['is_up'].notna())
            & (self.df['bid'].notna())
            & (self.df['ask'].notna())
            & (self.df['target'].notna())
            & (self.df['bid'] > 0)
            & (self.df['ask'] > 0)
            & (self.df['price'] > 0)
            & (self.df['target'] > 0)
        ].copy()

        price_vals = self.df['price'].astype(float).to_numpy()
        target_vals = self.df['target'].astype(float).to_numpy()
        self.df["delta"] = price_vals - target_vals
        self.df["percent"] = (price_vals - target_vals) / target_vals
        self.df["log_return"] = np.log(price_vals / target_vals)


        # self.df = self.df[
        #     (self.df['delta'] > -150)
        #     & (self.df['delta'] < 150)
        # ].copy()
    
        self.df['seconds_left'] = 900 - (self.df['timestamp'] - self.df['interval'])
        self.df['time'] = self.df['seconds_left'].astype(float) / 900.
                
        # Sort by timestamp to ensure proper rolling window calculation
        self.df = self.df.sort_values('timestamp').reset_index(drop=True)
        
        # Compute rolling sigma of log_return with window 900
        self.df['sigma'] = self.df['log_return'].rolling(window=60, min_periods=1).std()
        
        # Estimate z_score from log_return, sigma, and seconds_left
        # z_score = log_return / (sigma * sqrt(seconds_left / 900))
        # This scales the volatility by the time remaining
        time_factor = np.sqrt(self.df['seconds_left'] / 60.0)
        # Avoid division by zero
        sigma_scaled = self.df['sigma'].replace(0, np.nan)
        self.df['z_score'] = self.df['log_return'] / (sigma_scaled * time_factor)
        # Fill NaN values (where sigma was 0) with 0
        self.df['z_score'] = self.df['z_score'].fillna(0)
        
        # Use norm.cdf to get probability estimates
        self.df["prob_est"] = norm.cdf(self.df["z_score"])

    def _train_test_split(self, test_ratio: float = 0.2):
        df_sorted = self.df.sort_values('timestamp').reset_index(drop=True)
        split_idx = int(len(df_sorted) * (1 - test_ratio))
        split_idx = max(1, min(split_idx, len(df_sorted) - 1))
        self.train_df = df_sorted.iloc[:split_idx].copy()
        self.test_df = df_sorted.iloc[split_idx:].copy()

        self.df['label'] = self.df['is_up'].astype(int)
        self.train_df['label'] = self.train_df['is_up'].astype(int)
        self.test_df['label'] = self.test_df['is_up'].astype(int)

        # print(self.train_df.head())
        # print(self.test_df.head())

        return self.train_df, self.test_df

    def evaluate_strategy(self, df: pd.DataFrame, spread: float = 0.05, probability_column: str = 'probability'):
        if probability_column not in df.columns:
            raise ValueError(f"Dataframe must contain '{probability_column}' column for evaluation.")

        eval_df = df.dropna(subset=[probability_column, 'bid', 'is_up']).copy()
        eval_df['action'] = (eval_df[probability_column] - spread >= eval_df['bid'] and eval_df['bid'] < 0.4)

        trade_df = eval_df[eval_df['action']].copy()
        trade_df['revenue'] = trade_df['is_up'].astype(float)
        trade_df['cost'] = trade_df['bid']
        trade_df['pnl'] = trade_df['revenue'] - trade_df['cost']

        summary = {
            'num_rows': len(df),
            'num_trades': len(trade_df),
            'revenue': trade_df['revenue'].sum(),
            'cost': trade_df['cost'].sum(),
            'pnl': trade_df['pnl'].sum(),
            'margin': trade_df['pnl'].mean(),
        }

        return summary, trade_df

    def evaluate_model_metrics(self, df: pd.DataFrame, probability_column: str = 'probability', spread: float = 0.05):
        eval_df = df.dropna(subset=[probability_column, 'label']).copy()
        y_true = eval_df['label'].astype(int)
        probs = eval_df[probability_column].astype(float)
        y_pred = (probs - spread > eval_df['bid']).astype(int)

        # Calculate confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        # Confusion matrix format: [[TN, FP], [FN, TP]]
        # Handle different matrix sizes
        if cm.shape == (2, 2):
            TN, FP, FN, TP = cm.ravel()
        elif cm.shape == (1, 1):
            # Only one class present in predictions
            if y_pred[0] == 0:
                TN, FP, FN, TP = cm[0, 0], 0, 0, 0
            else:
                TN, FP, FN, TP = 0, 0, 0, cm[0, 0]
        else:
            # Fallback: try to extract values safely
            TN = cm[0, 0] if cm.shape[0] >= 1 and cm.shape[1] >= 1 else 0
            FP = cm[0, 1] if cm.shape[0] >= 1 and cm.shape[1] >= 2 else 0
            FN = cm[1, 0] if cm.shape[0] >= 2 and cm.shape[1] >= 1 else 0
            TP = cm[1, 1] if cm.shape[0] >= 2 and cm.shape[1] >= 2 else 0
        
        # Calculate metrics from confusion matrix
        total = TN + FP + FN + TP
        accuracy = (TN + TP) / total if total > 0 else 0.0
        precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        pnl = float('nan')
        num_trades = 0
        if {'bid', 'is_up'}.issubset(eval_df.columns):
            actions = (probs - spread > eval_df['bid'])
            trade_df = eval_df[actions].copy()
            if len(trade_df) > 0:
                trade_df['revenue'] = trade_df['is_up'].astype(float)
                trade_df['cost'] = trade_df['bid']
                pnl = float((trade_df['revenue'] - trade_df['cost']).sum())
                num_trades = len(trade_df)

        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'roc_auc': roc_auc_score(y_true, probs) if len(np.unique(y_true)) > 1 else float('nan'),
            'total_pnl': pnl,
            'num_trades': num_trades,
            'num_rows': len(eval_df),
            'confusion_matrix': cm.tolist(),
        }


def main():
    dataset = Dataset()
    train_df = dataset.train_df
    test_df = dataset.test_df
    dataset.show()


    return 
if __name__ == "__main__":
    main()