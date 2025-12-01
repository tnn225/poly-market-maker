import os
import pandas as pd
import numpy as np
from poly_market_maker.dataset import Dataset

BID_BUCKETS = 200  # Number of bins for bid values
SECONDS_LEFT_BUCKET_SIZE = 1
SECONDS_LEFT_BUCKETS = int(900 / SECONDS_LEFT_BUCKET_SIZE)  # 15 bins (0-60, 60-120, ..., 840-900)
TOP_PERCENTILE = 100  # Top 10% threshold (90th percentile)

class BidClassifier:
    def __init__(self):
        self.delta_thresholds = {}  # Key: (bid_bin, seconds_left_bin), Value: 90th percentile delta threshold
        self._bid_percentiles = None
        self._seconds_left_bins = None

    def fit(self, X, y):
        """
        Build a mapping of delta thresholds by bid and seconds_left.
        For each (bid, seconds_left) combination, calculate the 90th percentile of deltas.
        """
        # Create a temporary dataframe with required columns
        required_cols = ['delta', 'seconds_left', 'bid', 'interval']
        df = X[required_cols].copy()
        df['is_up'] = y
        
        threshold_data = {}
        for i, row in df.iterrows():
            # Use rounded values to avoid floating point precision issues
            key = (round(float(row['bid']), 2), int(row['seconds_left']))
            if key not in threshold_data:
                threshold_data[key] = {
                    'bid': row['bid'],
                    'seconds_left': row['seconds_left'],
                    'deltas': {},
                    'intervals': {},
                }
            threshold_data[key]['deltas'][row['delta']] = 1
            threshold_data[key]['intervals'][row['interval']] = 1
        
        for key, value in threshold_data.items():
            # Convert dict keys to list for percentile and other calculations
            deltas_list = list(value['deltas'].keys())
            value['threshold'] = np.percentile(deltas_list, TOP_PERCENTILE)
            value['count'] = len(value['deltas'])
            value['interval_count'] = len(value['intervals'])
            value['min_delta'] = min(deltas_list)
            value['max_delta'] = max(deltas_list)
            value['mean_delta'] = np.mean(deltas_list)
            # Keep deltas as a list for easier access later
            value['deltas'] = deltas_list

            self.delta_thresholds[key] = value

        print(f"Built delta threshold map with {len(self.delta_thresholds)} (bid, seconds_left) combinations")
        print(f"Sample thresholds (first 10):")
        for i, (key, value) in enumerate(list(self.delta_thresholds.items())[:100]):
            print(f"  {key}: threshold={value['threshold']:.2f}, count={value['count']}, min={value['min_delta']}, max={value['max_delta']}, mean={value['mean_delta']}, interval_count={value['interval_count']}")

    def should_buy(self, delta: float, bid: float, seconds_left: float) -> bool:
        """
        Determine if action should be 'buy' based on whether delta is in top 10% 
        for the given (bid, seconds_left) combination.
        
        Returns True if delta >= 90th percentile threshold for (bid, seconds_left), False otherwise.
        """
        if len(self.delta_thresholds) == 0:
            return False
        
        # Use rounded values to match the keys stored during fit
        key = (round(bid, 2), int(seconds_left))
        threshold_info = self.delta_thresholds.get(key)
        
        if threshold_info is None:
            # If no data for this combination, return False (conservative)
            return False
        
        threshold = threshold_info['threshold']
        return delta >= threshold

    def get_threshold(self, bid: float, seconds_left: float) -> float:
        """Get the 90th percentile delta threshold for a given (bid, seconds_left) combination"""
        # Use rounded values to match the keys stored during fit
        key = (round(bid, 2), int(seconds_left))
        threshold_info = self.delta_thresholds.get(key)
        
        if threshold_info is None:
            return None
        
        return threshold_info['threshold']

    def predict_proba(self, X):
        """
        Generate probability predictions for the input data.
        
        Parameters:
        -----------
        X : pandas.DataFrame or numpy.ndarray
            Input features with columns ['delta', 'seconds_left', 'bid']
        
        Returns:
        --------
        numpy.ndarray of shape (n_samples, 2)
            Probability predictions where:
            - Column 0: probability of class 0 (negative)
            - Column 1: probability of class 1 (positive)
        """
        if len(self.delta_thresholds) == 0:
            raise ValueError("Model has not been trained yet. Call fit() first.")
        
        # Handle both DataFrame and numpy array inputs
        if isinstance(X, pd.DataFrame):
            df = X[['delta', 'seconds_left', 'bid']].copy()
        elif isinstance(X, np.ndarray):
            # Assume columns are in order: delta, seconds_left, bid
            df = pd.DataFrame(X, columns=['delta', 'seconds_left', 'bid'])
        else:
            raise ValueError("X must be a pandas DataFrame or numpy array")
        
        probabilities = []
        for idx, row in df.iterrows():
            delta = row['delta']
            bid = row['bid']
            seconds_left = row['seconds_left']
            
            should_buy_flag = self.should_buy(delta, bid, seconds_left)
            # Probability of positive class: 1.0 if should_buy, otherwise use bid
            prob_positive = 1.0 if should_buy_flag else bid
            prob_positive = float(np.clip(prob_positive, 0.0, 1.0))
            prob_negative = 1.0 - prob_positive
            
            probabilities.append([prob_negative, prob_positive])
        
        return np.array(probabilities)

    def get_probability(self, price, target, seconds_left, bid, ask):
        """
        Generate a probability prediction based on whether delta is in top 10% for (bid, seconds_left).
        Returns 1.0 if should_buy, 0.0 otherwise.
        """
        if len(self.delta_thresholds) == 0:
            return 0.5
        
        if target <= 0 or price <= 0:
            raise ValueError("price and target must be positive numbers.")
        
        delta = price - target
        
        if seconds_left <= 0:
            return float(price >= target)
        
        should_buy_flag = self.should_buy(delta, bid, seconds_left)
        threshold = self.get_threshold(bid, seconds_left)
        
        probability = 1.0 if should_buy_flag else bid
        threshold_str = f"{threshold:.2f}" if threshold is not None else "N/A"
        print(f"price: {price}, delta: {delta:.2f}, seconds_left: {seconds_left}, bid: {bid}, ask: {ask}, "
              f"threshold: {threshold_str}, should_buy: {should_buy_flag}, probability: {probability}")
        
        return probability

def main():
    dataset = Dataset()
    train_df = dataset.train_df
    test_df = dataset.test_df

    # Ensure we have the required columns
    feature_cols = ['delta', 'seconds_left', 'bid', 'interval']
    if not all(col in train_df.columns for col in feature_cols):
        print(f"Missing columns. Available: {train_df.columns.tolist()}")
        return

    model = BidClassifier()
    model.fit(train_df[feature_cols], train_df['label'])

    # Make predictions
    prob = model.predict_proba(test_df[feature_cols])
    test_df['probability'] = prob[:, 1]
    ret = dataset.evaluate_model_metrics(test_df, probability_column='probability', spread=0.05)
    print(ret)
 
if __name__ == "__main__":
    main()
