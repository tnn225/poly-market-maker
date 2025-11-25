from datetime import datetime, timezone, timedelta
import os
import time
import pandas as pd
import numpy as np
import logging

import csv

from collections import deque

from lightgbm import LGBMRegressor

from scipy.stats import norm
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error

SPREAD = 0.05
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

        self.train_df['label'] = self.train_df['is_up'].astype(float) - self.train_df['bid'].astype(float)
        self.test_df['label'] = self.test_df['is_up'].astype(float) - self.test_df['bid'].astype(float)

        return self.train_df, self.test_df

    def evaluate_strategy(self, df: pd.DataFrame, spread: float = 0.05, probability_column: str = 'probability'):
        if probability_column not in df.columns:
            raise ValueError(f"Dataframe must contain '{probability_column}' column for evaluation.")

        eval_df = df.dropna(subset=[probability_column, 'bid', 'is_up']).copy()
        eval_df['action'] = (eval_df[probability_column] - spread > eval_df['bid']) & (eval_df['bid'] > 0)

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
        y_true = eval_df['label'].astype(float)
        y_pred = eval_df[probability_column].astype(float)

        pnl = float('nan')
        if {'bid', 'is_up'}.issubset(eval_df.columns):
            actions = (y_pred >= spread) & (eval_df['bid'] > 0)
            trade_df = eval_df[actions].copy()
            trade_df['revenue'] = trade_df['is_up'].astype(float)
            trade_df['cost'] = trade_df['bid']
            pnl = float((trade_df['revenue'] - trade_df['cost']).sum())

        rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
        mae = float(mean_absolute_error(y_true, y_pred))

        return {
            'rmse': rmse,
            'mae': mae,
            'total_pnl': pnl,
        }


def main():
    dataset = Dataset()

    train_df, test_df = dataset.train_test_split()

    # print(train_df.head())
    # print(test_df.head())

    # feature_cols = ['log_return', 'time', 'seconds_left']
    feature_cols = ['delta', 'time']

    print(train_df[feature_cols].isna().sum())
    print(train_df['label'].isna().sum())


    candidate_models = []

    # candidate_models.append(("LightGBM", LGBMClassifier(n_estimators=200, max_depth=-1, learning_rate=0.05, random_state=42,)))

    # candidate_models.append(("LGBMRegressor", LGBMRegressor(n_estimators=1000, max_depth=-1, learning_rate=0.01, random_state=42)))
    candidate_models.append(("RandomForestRegressor", RandomForestRegressor(n_estimators=500, max_depth=6, min_samples_split=50, random_state=42, n_jobs=-1)))
    # candidate_models.append(("GradientBoostingRegressor", GradientBoostingRegressor(n_estimators=1000, learning_rate=0.01, max_depth=6, random_state=42)))
    # candidate_models.append(("LinearRegression", LinearRegression()))
    # candidate_models.append(("MLPRegressor", MLPRegressor(hidden_layer_sizes=(100, 20), activation='relu', solver='adam', learning_rate_init=0.01, max_iter=10000, random_state=42)))

    model_results = []
    for name, model in candidate_models:
        model.fit(train_df[feature_cols], train_df['label'])
        preds = model.predict(test_df[feature_cols])
        preds = np.clip(preds, 0.0, 1.0)
        test_df[f'prob_{name}'] = preds
        metrics = dataset.evaluate_model_metrics(test_df, probability_column=f'prob_{name}', spread=SPREAD)
        metrics['model'] = name
        model_results.append(metrics)
        print(datetime.now(), metrics)

    results_df = pd.DataFrame(model_results).sort_values('total_pnl', ascending=False)
    best_model_name = results_df.iloc[0]['model']
    test_df['probability'] = test_df[f'prob_{best_model_name}']

    # summary, buy_df = dataset.evaluate_strategy(test_df, spread=SPREAD, probability_column='probability')
    # print(summary)
    # print(buy_df.head())

if __name__ == "__main__":
    main()