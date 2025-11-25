from datetime import datetime, timezone, timedelta
import os
import time
import pandas as pd
import numpy as np
import logging

import csv

from collections import deque

from scipy.stats import norm

VOL_WINDOW = 10000
logger = logging.getLogger(__name__)

class PredictionEngine:
    def __init__(self):
        self._delta_percentiles = None
        self._read_dates()
        self._add_target_and_is_up()
        self._create_table()

    def _read_dates(self):
        today = datetime.now()
        dates = [(today - timedelta(days=i)).strftime("%Y-%m-%d") for i in range(7)]
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
        self.df = self.df[(self.df['is_up'].notna()) & (self.df['target'].notna())].copy()

        self.df["delta"] = self.df["price"] - self.df["target"]
        
        self.df['seconds_left'] = 900 - (self.df['timestamp'] - self.df['interval'])

        self.df = self.df.drop(columns=['interval'])
        
        # Bin delta into 100 percentile bins using fixed percentile thresholds
        delta_values = self.df['delta'].dropna()
        if len(delta_values) > 0:
            delta_percentiles = np.percentile(delta_values, np.linspace(0, 100, 101))
            delta_percentiles = pd.Series(delta_percentiles).drop_duplicates().values
            if len(delta_percentiles) > 1:
                delta_percentiles[0] = delta_values.min() - 1e-10
                delta_percentiles[-1] = delta_values.max() + 1e-10
                # Sort percentiles for faster binary search lookup
                self._delta_percentiles = np.sort(delta_percentiles)
                
                try:
                    self.df['delta_bin'] = pd.cut(
                        self.df['delta'], 
                        bins=delta_percentiles,
                        labels=False,
                        include_lowest=True
                    )
                except (ValueError, IndexError):
                    unique_deltas = delta_values.nunique()
                    n_bins = min(100, unique_deltas)
                    if n_bins > 1:
                        self.df['delta_bin'] = pd.qcut(
                            self.df['delta'], 
                            q=n_bins, 
                            labels=False, 
                            duplicates='drop'
                        )
                    else:
                        self.df['delta_bin'] = 0
            else:
                self.df['delta_bin'] = 0
        else:
            self.df['delta_bin'] = 0
        
        # Bin seconds_left into 10 bins
        self.df['seconds_left_bin'] = pd.cut(
            self.df['seconds_left'],
            bins=10,
            labels=False,
            include_lowest=True
        )
        
        # Fill NaN values
        self.df['delta_bin'] = self.df['delta_bin'].fillna(0).astype(int)
        self.df['seconds_left_bin'] = self.df['seconds_left_bin'].fillna(0).astype(int)

    def _create_table(self):
        """Create table grouped by seconds_left_bin and delta_bin with indexed lookup"""
        # Initialize table_index as empty dict
        self.table_index = {}
        
        if self.df.empty or 'is_up' not in self.df.columns:
            self.table = pd.DataFrame(columns=['seconds_left_bin', 'delta_bin', 'count', 'mean'])
            return
        
        grouped = self.df.groupby(['seconds_left_bin', 'delta_bin']).agg(
            count=('is_up', 'count'),
            mean=('is_up', 'mean'),
        ).round(4)
        grouped = grouped.reset_index()
        self.table = grouped
        
        # Create indexed lookup dictionary for O(1) access
        # Key: (seconds_left_bin, delta_bin), Value: mean
        for _, row in self.table.iterrows():
            key = (int(row['seconds_left_bin']), int(row['delta_bin']))
            self.table_index[key] = row['mean']

        # Persist aggregated odds to CSV for downstream analysis in pivot form
        pivot_df = grouped.pivot(index='seconds_left_bin', columns='delta_bin', values='mean').sort_index()
        pivot_df = pivot_df.sort_index(axis=1)

        os.makedirs("./data", exist_ok=True)
        pivot_df.to_csv("./data/odds.csv")

    def get_delta_bin(self, delta: float) -> int:
        """Get delta bin number for a given delta value using binary search"""
        if self._delta_percentiles is None or len(self._delta_percentiles) < 2:
            return 0
        
        percentiles = self._delta_percentiles
        
        # Handle values outside the range
        if delta < percentiles[0]:
            return 0
        if delta >= percentiles[-1]:
            return len(percentiles) - 2
        
        # Use binary search for O(log n) lookup
        idx = np.searchsorted(percentiles, delta, side='right') - 1
        # Ensure idx is within valid range
        idx = max(0, min(idx, len(percentiles) - 2))
        return idx

    def get_seconds_left_bin(self, seconds_left: float) -> int:
        """Get seconds_left bin number (0-9)"""
        if seconds_left < 0:
            return 0
        if seconds_left >= 900:
            return 9
        return int(seconds_left // 90)

    def get_probability(self, price, target, seconds_left):
        delta = price - target
        
        if seconds_left <= 0:
            return float(price >= target)
        
        delta_bin = self.get_delta_bin(delta)
        seconds_left_bin = self.get_seconds_left_bin(seconds_left)
        
        # Use indexed lookup for O(1) access
        key = (seconds_left_bin, delta_bin)
        ret = self.table_index.get(key, 0.5)  # Default to 0.5 if not found
        
        # print(f"price: {price}, target: {target}, seconds_left: {seconds_left}: {ret}")
        return ret


def main():
    today = datetime.now()
    dates = [(today - timedelta(days=i)).strftime("%Y-%m-%d") for i in range(7)]
    dates.reverse()
    
    prediction_engine = PredictionEngine()
    
    

if __name__ == "__main__":
    main()