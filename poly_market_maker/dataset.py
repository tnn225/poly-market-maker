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
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)

SPREAD = 0.01
DAYS = 7
logger = logging.getLogger(__name__)

class Dataset:
    def __init__(self):
        self._delta_percentiles = None
        self._read_dates()
        self._add_target_and_is_up()

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
            & (self.df['price'] > 0)
            & (self.df['target'] > 0)
        ].copy()

        price_vals = self.df['price'].astype(float).to_numpy()
        target_vals = self.df['target'].astype(float).to_numpy()
        self.df["delta"] = price_vals - target_vals
        self.df["percent"] = (price_vals - target_vals) / target_vals
        self.df["log_return"] = np.log(price_vals / target_vals)
        
        
        self.df['seconds_left'] = 900 - (self.df['timestamp'] - self.df['interval'])
        self.df['time'] = self.df['seconds_left'] / 900.
        self.df['time'] = self.df['time'].astype(float)

        self.df = self.df.drop(columns=['interval'])

    def train_test_split(self, test_ratio: float = 0.2):
        df_sorted = self.df.sort_values('timestamp').reset_index(drop=True)
        split_idx = int(len(df_sorted) * (1 - test_ratio))
        split_idx = max(1, min(split_idx, len(df_sorted) - 1))
        self.train_df = df_sorted.iloc[:split_idx].copy()
        self.test_df = df_sorted.iloc[split_idx:].copy()

        self.train_df['label'] = self.train_df['is_up'].astype(int)
        self.test_df['label'] = self.test_df['is_up'].astype(int)

        print(self.train_df.head())
        print(self.test_df.head())

        return self.train_df, self.test_df

    def evaluate_strategy(self, df: pd.DataFrame, spread: float = 0.05, probability_column: str = 'probability'):
        if probability_column not in df.columns:
            raise ValueError(f"Dataframe must contain '{probability_column}' column for evaluation.")

        eval_df = df.dropna(subset=[probability_column, 'bid', 'is_up']).copy()
        eval_df['action'] = (eval_df[probability_column] - spread >= eval_df['bid']) & (eval_df['bid'] > 0)

        buy_df = eval_df[eval_df['action']].copy()
        buy_df['revenue'] = buy_df['is_up'].astype(float)
        buy_df['cost'] = buy_df['bid']
        buy_df['pnl'] = buy_df['revenue'] - buy_df['cost']

        summary = {
            'total_rows': len(df),
            'eligible_rows': len(eval_df),
            'buy_trades': len(buy_df),
            'total_revenue': buy_df['revenue'].sum(),
            'total_cost': buy_df['cost'].sum(),
            'total_pnl': buy_df['pnl'].sum(),
            'avg_pnl_per_trade': buy_df['pnl'].mean() if not buy_df.empty else 0.0,
        }

        return summary, buy_df

    def evaluate_model_metrics(self, df: pd.DataFrame, probability_column: str = 'probability', threshold: float = 0.5, spread: float = 0.1):
        eval_df = df.dropna(subset=[probability_column, 'label']).copy()
        y_true = eval_df['label'].astype(int)
        probs = eval_df[probability_column].astype(float)
        preds = (probs >= threshold).astype(int)

        pnl = float('nan')
        if {'bid', 'is_up'}.issubset(eval_df.columns):
            actions = (probs - spread > eval_df['bid']) & (eval_df['bid'] > 0)
            trade_df = eval_df[actions].copy()
            trade_df['revenue'] = trade_df['is_up'].astype(float)
            trade_df['cost'] = trade_df['bid']
            pnl = float((trade_df['revenue'] - trade_df['cost']).sum())

        return {
            'accuracy': accuracy_score(y_true, preds),
            'precision': precision_score(y_true, preds, zero_division=0),
            'recall': recall_score(y_true, preds, zero_division=0),
            'f1_score': f1_score(y_true, preds, zero_division=0),
            'roc_auc': roc_auc_score(y_true, probs) if len(np.unique(y_true)) > 1 else float('nan'),
            'total_pnl': pnl,
        }


def main():
    dataset = Dataset()

    train_df, test_df = dataset.train_test_split()

    # print(train_df.head())
    # print(test_df.head())

    feature_cols = ['delta', 'percent', 'log_return', 'time', 'seconds_left', 'bid', 'ask']

    # feature_cols = ['delta', 'log_return', 'time', 'seconds_left',]

    candidate_models = []

    # candidate_models.append(("LGBMClassifier", LGBMClassifier(n_estimators=1000, max_depth=-1, learning_rate=0.01, random_state=42,)))
    candidate_models.append(("RandomForestClassifier", RandomForestClassifier(n_estimators=1000, max_depth=3, min_samples_split=50, random_state=42, n_jobs=-1)))
    #candidate_models.append(("GradientBoostingClassifier", GradientBoostingClassifier(n_estimators=1000, learning_rate=0.01, max_depth=3, random_state=42)))
    #candidate_models.append(("LogisticRegression", LogisticRegression(max_iter=1000, C=0.5, solver='lbfgs')))
    #candidate_models.append(("MLPClassifier", MLPClassifier(hidden_layer_sizes=(1000, 100, 10), activation='relu', solver='adam', learning_rate_init=0.01, max_iter=10000, random_state=42)))

    model_results = []
    for name, model in candidate_models:
        model.fit(train_df[feature_cols], train_df['label'])
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(test_df[feature_cols])[:, 1]
        else:
            probs = model.predict(test_df[feature_cols])
            probs = np.clip(probs, 0.0, 1.0)
        test_df[f'prob_{name}'] = probs
        metrics = dataset.evaluate_model_metrics(test_df, probability_column=f'prob_{name}', spread=SPREAD)
        metrics['model'] = name
        model_results.append(metrics)
        print(datetime.now(), metrics)


    return 

    results_df = pd.DataFrame(model_results).sort_values('total_pnl', ascending=False)
    best_model_name = results_df.iloc[0]['model']
    test_df['probability'] = test_df[f'prob_{best_model_name}']

    false_positive_actions = test_df[
        (test_df['probability'] - SPREAD >= test_df['bid'])
        & (test_df['bid'] > 0)
        & (test_df['is_up'] == 0)
    ].copy()
    false_positive_actions['minutes_left'] = (false_positive_actions['seconds_left'] // 60).astype(int)
    false_positive_actions = false_positive_actions[['probability', 'bid', 'delta', 'seconds_left', 'minutes_left']].sort_values('probability')
    if not false_positive_actions.empty:
        print("False positive actions (probability vs bid):")
        # print(false_positive_actions.to_string(index=False))
        bucket_counts = false_positive_actions.groupby('minutes_left').size().reset_index(name='count').sort_values('minutes_left')
        print("False positive counts by minutes_left bucket:")
        print(bucket_counts.to_string(index=False))

        try:
            false_positive_actions['bid_bucket'] = pd.qcut(
                false_positive_actions['bid'],
                q=10,
                duplicates='drop'
            )
        except ValueError:
            false_positive_actions['bid_bucket'] = pd.cut(
                false_positive_actions['bid'],
                bins=10,
                include_lowest=True
            )

        bid_bucket_counts = (
            false_positive_actions
            .groupby('bid_bucket')
            .size()
            .reset_index(name='count')
            .sort_values('bid_bucket')
        )
        print("False positive counts by bid bucket (10 buckets):")
        print(bid_bucket_counts.to_string(index=False))
    else:
        print("No false positive actions found.")

    # summary, buy_df = dataset.evaluate_strategy(test_df, spread=SPREAD, probability_column='probability')
    # print(summary)
    # print(buy_df.head())

if __name__ == "__main__":
    main()