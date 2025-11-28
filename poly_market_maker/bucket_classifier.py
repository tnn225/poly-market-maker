import os
import pandas as pd
import numpy as np
from poly_market_maker.dataset import Dataset

class BucketClassifier:
    def __init__(self):
        self.table_index = {}
        self.table = None
        self._delta_percentiles = None

    def fit(self, X, y):
        # Create a temporary dataframe for binning
        df = X[['delta', 'seconds_left']].copy()
        df['is_up'] = y
        
        # Bin delta into 100 percentile bins using fixed percentile thresholds
        delta_values = df['delta'].dropna()
        if len(delta_values) > 0:
            delta_percentiles = np.percentile(delta_values, np.linspace(0, 100, 101))
            delta_percentiles = pd.Series(delta_percentiles).drop_duplicates().values
            if len(delta_percentiles) > 1:
                delta_percentiles[0] = delta_values.min() - 1e-10
                delta_percentiles[-1] = delta_values.max() + 1e-10
                # Sort percentiles for faster binary search lookup
                self._delta_percentiles = np.sort(delta_percentiles)
                
                try:
                    df['delta_bin'] = pd.cut(
                        df['delta'], 
                        bins=delta_percentiles,
                        labels=False,
                        include_lowest=True
                    )
                except (ValueError, IndexError):
                    unique_deltas = delta_values.nunique()
                    n_bins = min(100, unique_deltas)
                    if n_bins > 1:
                        df['delta_bin'] = pd.qcut(
                            df['delta'], 
                            q=n_bins, 
                            labels=False, 
                            duplicates='drop'
                        )
                    else:
                        df['delta_bin'] = 0
            else:
                df['delta_bin'] = 0
        else:
            df['delta_bin'] = 0
        
        # Bin seconds_left into 10 bins
        df['seconds_left_bin'] = pd.cut(
            df['seconds_left'],
            bins=15,
            labels=False,
            include_lowest=True
        )
        
        # Fill NaN values
        df['delta_bin'] = df['delta_bin'].fillna(0).astype(int)
        df['seconds_left_bin'] = df['seconds_left_bin'].fillna(0).astype(int)
        
        # Group by (seconds_left_bin, delta_bin) and calculate mean is_up
        grouped = df.groupby(['seconds_left_bin', 'delta_bin']).agg(
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
            return 15
        return int(seconds_left // 60)

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
        prob = self.table_index.get(key, 0.5)  # Default to 0.5 if not found
        
        return float(np.clip(prob, 0.0, 1.0))

def main():
    dataset = Dataset()
    train_df = dataset.train_df
    test_df = dataset.test_df

    model = BucketClassifier()
    model.fit(train_df[['delta', 'seconds_left']], train_df['label'])

    prob = model.predict_proba(test_df[['delta', 'seconds_left']])
    print(f"Predictions shape: {prob.shape}")
    print(f"First 5 predictions: {prob[:5]}")
    
    # Test single prediction
    test_prob = model.get_probability(87576.878879, 87628.002972, 686, 0.4, 0.41)
    print(f"Single prediction: {test_prob}")

if __name__ == "__main__":
    main()