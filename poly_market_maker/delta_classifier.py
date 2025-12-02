from math import floor
import os
import time

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from poly_market_maker.dataset import Dataset

SPREAD = 0.05

class DeltaClassifier:
    def __init__(self):
        self.delta_thresholds = None

    def fit(self, X, y):
        # Create a temporary dataframe with required columns
        required_cols = ['delta', 'seconds_left', 'bid']
        df = X[required_cols].copy()
        df['is_up'] = y

        count = {} 
        up_count = {}
        down_count = {}
        rate = {} 

        for index, row in df.iterrows():
            seconds_left = row['seconds_left']
            delta = row['delta']
            if seconds_left not in count:
                count[seconds_left] = {}
                up_count[seconds_left] = {}
                down_count[seconds_left] = {}
                rate[seconds_left] = {}
            if delta not in count[seconds_left]:
                count[seconds_left][delta] = 0
                up_count[seconds_left][delta] = 0
                down_count[seconds_left][delta] = 0
                rate[seconds_left][delta] = 0
            count[seconds_left][delta] += 1 

            # Use 'is_up' if available, otherwise 'label'
            is_up_value = row.get('is_up', row.get('label', 0))
            if is_up_value:
                up_count[seconds_left][delta] += 1
            else:
                down_count[seconds_left][delta] += 1

        for seconds_left in count:
            total_down_count = sum(down_count[seconds_left].values())

            up = 0
            down = 0
            deltas = sorted(count[seconds_left].keys())
            for delta in deltas:
                up += up_count[seconds_left][delta]
                down += down_count[seconds_left][delta]
                rate[seconds_left][delta] = up / (up + total_down_count - down) if up > 0 else 0.0

        delta_thresholds = {}

        for seconds_left in rate:
            delta_thresholds[seconds_left] = {}
        
            deltas = sorted(rate[seconds_left].keys())
            
            for delta in deltas:
                rate_value = round(floor(rate[seconds_left][delta] * 100) / 100, 2)
                if rate_value not in delta_thresholds[seconds_left]:
                    delta_thresholds[seconds_left][rate_value] = delta

        self.delta_thresholds = delta_thresholds

    def get_up(self, seconds_left: float, delta: float, bid: float) -> float:
        if self.delta_thresholds is None or len(self.delta_thresholds) == 0:
            return bid

        if seconds_left not in self.delta_thresholds:
            return bid 

        best = 0.01 
        bid = 0.01
        while bid <= 1:
            bid = round(bid , 2)
            if bid in self.delta_thresholds[seconds_left]:
                if delta >= self.delta_thresholds[seconds_left][bid]:
                    best = bid
                else:
                    break 
            bid += 0.01 
        return best

    def should_buy(self, delta: float, bid: float, seconds_left: float) -> bool:
        if self.delta_thresholds is None or len(self.delta_thresholds) == 0:
            return False

        bid = round(bid, 2)
        
        if seconds_left not in self.delta_thresholds or bid not in self.delta_thresholds[seconds_left]:
            return False
        
        return delta >= self.delta_thresholds[seconds_left][bid]

    def predict_proba(self, X):
        if self.delta_thresholds is None or len(self.delta_thresholds) == 0:
            raise ValueError("Model has not been trained yet. Call fit() first.")
        
        # Create a temporary dataframe with required columns
        required_cols = ['delta', 'seconds_left', 'bid']
        df = X[required_cols].copy()
        
        probabilities = []
        for idx, row in df.iterrows():
            delta = row['delta']
            bid = row['bid'] 
            seconds_left = row['seconds_left']

            up = self.get_up(seconds_left, delta, bid)
            down = 1 - up

            prob_positive = up
            prob_negative = down

            prob_positive = float(np.clip(prob_positive, 0.0, 1.0))
            prob_negative = float(np.clip(prob_negative, 0.0, 1.0))

            probabilities.append([prob_negative, prob_positive])
        
        return np.array(probabilities)

    def get_probability(self, price, target, seconds_left, bid, ask):
        if self.delta_thresholds is None or len(self.delta_thresholds) == 0:
            raise ValueError("Model has not been trained yet. Call fit() first.")
        
        if target <= 0 or price <= 0:
            raise ValueError("price and target must be positive numbers.")

        bid = round(bid, 2)
        delta = price - target
        
        if seconds_left <= 0:
            return float(price >= target)

        up = self.get_up(seconds_left, delta, bid)
        return up 

    def plot_delta_thresholds(self, delta_thresholds, RATES):
        """
        Plot delta_thresholds by seconds_left and RATES.
        
        Parameters:
        -----------
        delta_thresholds : dict
            Dictionary with structure delta_thresholds[seconds_left][rate] = delta
        RATES : list
            List of rate values
        """
        # Convert to DataFrame for easier plotting
        data = []
        for seconds_left in sorted(delta_thresholds.keys()):
            for rate in RATES:
                if rate in delta_thresholds[seconds_left]:
                    data.append({
                        'seconds_left': seconds_left,
                        'rate': rate,
                        'delta': delta_thresholds[seconds_left][rate]
                    })
        
        if not data:
            print("No data to plot.")
            return
        
        df = pd.DataFrame(data)
        
        # Filter to only include seconds_left < 100
        df = df[df['seconds_left'] < 100]
        
        # Sort by seconds_left (ascending for x-axis)
        df = df.sort_values('seconds_left')
        
        # Create figure
        fig, ax = plt.subplots(1, 1, figsize=(14, 10))
        
        # Plot lines for each rate
        for rate in RATES:
            rate_data = df[df['rate'] == rate].sort_values('seconds_left')
            if len(rate_data) > 0:
                ax.plot(rate_data['seconds_left'], rate_data['delta'], 
                       marker='o', linewidth=2, markersize=4, label=f'Rate = {rate:.2f}')
        
        ax.set_xlabel('seconds_left', fontsize=12)
        ax.set_ylabel('delta_threshold', fontsize=12)
        ax.set_title('Delta Thresholds by Seconds Left and Rate', fontsize=14)
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        return df     

def main():
    dataset = Dataset()
    train_df = dataset.train_df
    test_df = dataset.test_df
 
    feature_cols = ['delta', 'seconds_left', 'bid', 'interval']
    model = DeltaClassifier()
    model.fit(train_df[feature_cols], train_df['label'])

    prob = model.predict_proba(test_df[feature_cols])
    test_df['probability'] = prob[:, 1]
    ret = dataset.evaluate_model_metrics(test_df, probability_column='probability', spread=0.05)
    print(ret)

if __name__ == "__main__":
    main()