from math import floor
import os
import time

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from poly_market_maker.dataset import Dataset

SPREAD = 0.05

class DeltaClassifier:
    def __init__(self):
        self.delta_thresholds = None
        self.df = {}
        self.models = {}  # Store models for each seconds_left

    def fit(self, X, y):
        # Create a temporary dataframe with required columns
        required_cols = ['delta', 'seconds_left', 'bid']
        df = X[required_cols].copy()
        df['label'] = y.astype(int)

        # Store dataframe filtered by seconds_left
        self.df = {}
        for seconds_left in df['seconds_left'].unique():
            self.df[seconds_left] = df[df['seconds_left'] == seconds_left].copy()

        # Fit a model for each seconds_left using delta and label
        self.models = {}
        for seconds_left in self.df:
            print(f"Fitting model for seconds_left: {seconds_left}")
            df_group = self.df[seconds_left]
            
            # Extract features (delta) and labels
            X_group = df_group[['delta']].values
            y_group = df_group['label'].values
            
            # Only fit if we have enough data
            if len(X_group) > 1 and len(np.unique(y_group)) > 1:
                # Fit logistic regression
                model = LogisticRegression()
                model.fit(X_group, y_group)
                self.models[seconds_left] = model
            else:
                # Store None if not enough data
                self.models[seconds_left] = None

    def predict_proba(self, X):
        if len(self.models) == 0:
            raise ValueError("Model has not been trained yet. Call fit() first.")
        
        # Create a temporary dataframe with required columns
        required_cols = ['delta', 'seconds_left', 'bid']
        df = X[required_cols].copy()
        
        probabilities = []
        for idx, row in df.iterrows():
            # print(f"Predicting for seconds_left: {row['seconds_left']} delta: {row['delta']} bid: {row['bid']}")
            seconds_left = row['seconds_left']
            delta = row['delta']
            bid = row['bid']
            
            # Check if model exists for this seconds_left
            if seconds_left in self.models and self.models[seconds_left] is not None:
                # Use the model to predict probability
                model = self.models[seconds_left]
                delta_array = np.array([[delta]])
                prob = model.predict_proba(delta_array)[0]
                prob_positive = prob[1]  # Probability of class 1 (positive)
                prob_negative = prob[0]  # Probability of class 0 (negative)
            else:
                # Fallback to bid if no model available
                prob_positive = bid
                prob_negative = 1.0 - bid
            
            prob_positive = float(np.clip(prob_positive, 0.0, 1.0))
            prob_negative = float(np.clip(prob_negative, 0.0, 1.0))
            
            probabilities.append([prob_negative, prob_positive])
        
        return np.array(probabilities)

    def get_probability(self, price, target, seconds_left, bid, ask):
        if len(self.models) == 0:
            raise ValueError("Model has not been trained yet. Call fit() first.")
        
        if target <= 0 or price <= 0:
            raise ValueError("price and target must be positive numbers.")

        delta = price - target
        
        if seconds_left <= 0:
            return float(price >= target)

        # Check if model exists for this seconds_left
        if seconds_left in self.models and self.models[seconds_left] is not None:
            # Use the model to predict probability
            model = self.models[seconds_left]
            delta_array = np.array([[delta]])
            prob = model.predict_proba(delta_array)[0]
            return float(prob[1])  # Probability of class 1 (positive)
        else:
            # Fallback to bid if no model available
            return float(bid)

    def plot_delta_by_seconds_left(self, SECONDS_LEFT_LIST=None):
        """
        Plot delta vs seconds_left for specified seconds_left values, colored by label.
        
        Parameters:
        -----------
        SECONDS_LEFT_LIST : list, optional
            List of seconds_left values to plot. Defaults to [1, 60, 450, 840]
        """
        if len(self.df) == 0:
            print("No data available. Call fit() first.")
            return
        
        if SECONDS_LEFT_LIST is None:
            SECONDS_LEFT_LIST = [1, 60, 450, 840]
        
        # Collect data for the specified seconds_left values
        plot_data = []
        for seconds_left in SECONDS_LEFT_LIST:
            # Find the closest seconds_left in self.df
            if seconds_left in self.df:
                df_group = self.df[seconds_left]
                for idx, row in df_group.iterrows():
                    plot_data.append({
                        'seconds_left': seconds_left,
                        'delta': row['delta'],
                        'label': row['label']
                    })
            else:
                # Find closest seconds_left value
                available_seconds = list(self.df.keys())
                if available_seconds:
                    closest = min(available_seconds, key=lambda x: abs(x - seconds_left))
                    if abs(closest - seconds_left) <= 5:  # Within 5 seconds
                        df_group = self.df[closest]
                        for idx, row in df_group.iterrows():
                            plot_data.append({
                                'seconds_left': seconds_left,  # Use requested value for plotting
                                'delta': row['delta'],
                                'label': row['label']
                            })
        
        if not plot_data:
            print(f"No data found for seconds_left values: {SECONDS_LEFT_LIST}")
            return
        
        df_plot = pd.DataFrame(plot_data)
        
        # Create figure
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        
        # Separate data by label
        label_0 = df_plot[df_plot['label'] == 0]
        label_1 = df_plot[df_plot['label'] == 1]
        
        # Plot points colored by label
        if len(label_0) > 0:
            ax.scatter(label_0['seconds_left'], label_0['delta'], 
                      c='red', marker='o', alpha=0.6, s=30, label='Label = 0')
        
        if len(label_1) > 0:
            ax.scatter(label_1['seconds_left'], label_1['delta'], 
                      c='green', marker='s', alpha=0.6, s=30, label='Label = 1')
        
        ax.set_xlabel('seconds_left', fontsize=12)
        ax.set_ylabel('delta', fontsize=12)
        ax.set_title('Delta vs Seconds Left (colored by label)', fontsize=14)
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        # Set x-axis ticks to show only the requested seconds_left values
        ax.set_xticks(SECONDS_LEFT_LIST)
        ax.set_xticklabels([str(sl) for sl in SECONDS_LEFT_LIST])
        
        plt.tight_layout()
        plt.show()
        
        return df_plot

def main():
    dataset = Dataset()
    train_df = dataset.train_df
    test_df = dataset.test_df
 
    feature_cols = ['delta', 'seconds_left', 'bid', 'interval']
    model = DeltaClassifier()
    model.fit(train_df[feature_cols], train_df['label'])

    # Plot delta vs seconds_left for specified values
    # SECONDS_LEFT_LIST = [1, 60, 450, 840]
    # model.plot_delta_by_seconds_left(SECONDS_LEFT_LIST)

    prob = model.predict_proba(test_df[feature_cols])
    test_df['probability'] = prob[:, 1]
    ret = dataset.evaluate_model_metrics(test_df, probability_column='probability', spread=0.05)
    print(ret)

if __name__ == "__main__":
    main()