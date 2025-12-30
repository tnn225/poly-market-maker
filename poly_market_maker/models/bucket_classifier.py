import os
import pandas as pd
import numpy as np
from poly_market_maker.dataset import Dataset

SPREAD = 0.1
DELTA_BUCKETS = 20
SECONDS_LEFT_BUCKET_SIZE = 180
SECONDS_LEFT_BUCKETS = int(900 / SECONDS_LEFT_BUCKET_SIZE)  # every 15 seconds = 60 bins
MAX_SECONDS_LEFT_BIN = SECONDS_LEFT_BUCKETS 

class BucketClassifier:
    def __init__(self):
        self.table_index = {}
        self.table = None
        self._delta_percentiles = None

    def fit(self, X, y):
        # Create a temporary dataframe for binning
        df = X[['delta', 'seconds_left']].copy()
        df['is_up'] = y

        df['interval'] = X['interval']
        df['timestamp'] = X['timestamp']

        # Bin delta into 100 percentile bins using fixed percentile thresholds
        delta_values = df['delta'].dropna()
        if len(delta_values) > 0:
            delta_percentiles = np.percentile(delta_values, np.linspace(0, 100, DELTA_BUCKETS))
            delta_percentiles = pd.Series(delta_percentiles).drop_duplicates().values
            if len(delta_percentiles) > 1:
                delta_percentiles[0] = delta_values.min() - 1e-10
                delta_percentiles[-1] = delta_values.max() + 1e-10
                # Sort percentiles for faster binary search lookup
                self._delta_percentiles = np.sort(delta_percentiles)
                df['delta_bin'] = pd.cut(
                    df['delta'], 
                    bins=delta_percentiles,
                    labels=False,
                    include_lowest=True
                )

        # Bin seconds_left into fixed 15-second buckets (0-15, 15-30, ..., 885-900)
        seconds_left_bins = np.arange(0, 901, SECONDS_LEFT_BUCKET_SIZE)
        df['seconds_left_bin'] = pd.cut(
            df['seconds_left'],
            bins=seconds_left_bins,
            labels=False,
            include_lowest=True
        )
        
        # Fill NaN values
        df['delta_bin'] = df['delta_bin'].fillna(0).astype(int)
        df['seconds_left_bin'] = df['seconds_left_bin'].fillna(0).astype(int)

        df_interval = (
            df
            .sort_values("timestamp")          # or seconds_left ascending
            .groupby("interval", as_index=False)
            .tail(1)
        )
        
        # Group by (seconds_left_bin, delta_bin) and calculate mean is_up
        grouped = df_interval.groupby(['seconds_left_bin', 'delta_bin']).agg(
            count=('is_up', 'count'),
            mean=('is_up', 'mean'),
        )
        
        grouped = grouped.round(4)
        grouped = grouped.reset_index()
        self.table = grouped
        
        # Create indexed lookup dictionary for O(1) access
        # Key: (seconds_left_bin, delta_bin), Value: mean
        for _, row in self.table.iterrows():
            key = (int(row['seconds_left_bin']), int(row['delta_bin']))
            self.table_index[key] = row['mean']

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
        """Get seconds_left bin number (0 to SECONDS_LEFT_BUCKETS-1)"""
        if seconds_left < 0:
            return 0
        if seconds_left >= 900:
            return MAX_SECONDS_LEFT_BIN - 1
        return int(seconds_left // SECONDS_LEFT_BUCKET_SIZE)

    def predict_proba(self, X, batch_size=64):
        """
        Generate predictions for the test dataset.

        Returns a numpy array of predictions in sklearn format (n_samples, 2).
        """
        if self.table is None:
            raise ValueError("Model has not been trained yet. Call fit() first.")
 
        probabilities = []
        for idx, row in X.iterrows():
            delta = row['delta']
            seconds_left = row['seconds_left']
            
            delta_bin = self.get_delta_bin(delta)
            seconds_left_bin = self.get_seconds_left_bin(seconds_left)
            
            # Use indexed lookup for O(1) access
            key = (seconds_left_bin, delta_bin)
            prob = self.table_index.get(key, 0.5)  # Default to 0.5 if not found
            probabilities.append(prob)
        
        prob_positive = np.array(probabilities)
        prob_negative = 1 - prob_positive
        return np.column_stack([prob_negative, prob_positive])

    def get_probability(self, price, target, seconds_left, bid, ask):
        """Generate a single probability prediction for the provided snapshot."""
        if self.table is None:
            raise ValueError("Model has not been trained yet. Call fit() first.")

        if target <= 0 or price <= 0:
            raise ValueError("price and target must be positive numbers.")

        delta = price - target
        
        if seconds_left <= 0:
            return float(price >= target)
        
        delta_bin = self.get_delta_bin(delta)
        seconds_left_bin = self.get_seconds_left_bin(seconds_left)
        
        # Use indexed lookup for O(1) access
        key = (seconds_left_bin, delta_bin)
        probability = self.table_index.get(key, bid)  # Default to bid if not found
        print(f"price: {price}, delta: {price - target}, seconds_left: {seconds_left}, bid: {bid}, ask: {ask}, probability: {probability}")

        return float(np.clip(probability, 0.0, 1.0))

def show_pnl_by_interval(df):
    df = df[(df['bid'] + SPREAD <= df['probability'])]
    df['pnl'] = df['is_up'] - df['bid']
    grouped = df.groupby('interval')['pnl'].agg(['sum', 'count'])

    total_pnl = 0
    volume = 0
    num_intervals = 0
    for interval, row in grouped.iterrows():
        pnl = row['sum']
        count = row['count']
        print(f"Interval: {interval}, PnL: {pnl}, Count: {count}")
        total_pnl += pnl
        volume += count
        num_intervals += 1
    print(f"Total PnL: {total_pnl} volume {volume} pnl/volume {total_pnl/volume:.2f} num_intervals {num_intervals} pnl/interval {total_pnl/num_intervals:.2f}")

def main():
    dataset = Dataset()
    train_df = dataset.train_df
    test_df = dataset.test_df

    model = BucketClassifier()
    # Include interval if available for unique counting
    feature_cols = ['delta', 'seconds_left', 'interval', 'timestamp']
    model.fit(train_df[feature_cols], train_df['label'])

    test_df['probability'] = model.predict_proba(test_df[['delta', 'seconds_left']])[:, 1]

    show_pnl_by_interval(test_df)

if __name__ == "__main__":
    main()

    