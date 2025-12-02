import os
import time

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from poly_market_maker.dataset import Dataset

SPREAD = 0.1
BID_BUCKETS = 200  # Number of bins for bid values
SECONDS_LEFT_BUCKET_SIZE = 1
SECONDS_LEFT_BUCKETS = int(900 / SECONDS_LEFT_BUCKET_SIZE)  # 15 bins (0-60, 60-120, ..., 840-900)
TOP_PERCENTILE = 50  # Top 50% threshold (90th percentile)

class BidClassifier:
    def __init__(self):
        self.delta_thresholds = {}  # Key: (bid_bin, seconds_left_bin), Value: 90th percentile delta threshold
        self._bid_percentiles = None
        self._seconds_left_bins = None
        self.linear_models = {}  # Key: bid (rounded), Value: dict with 'slope' and 'intercept' for linear model

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
        
        # Fit linear models for each bid value
        self._fit_linear_models()

    def _detect_outliers(self, residuals, method='iqr', multiplier=1.5):
        """
        Detect outliers in residuals using IQR or z-score method.
        
        Parameters:
        -----------
        residuals : numpy.ndarray
            Array of residuals from initial model fit
        method : str
            'iqr' for Interquartile Range method, 'zscore' for z-score method
        multiplier : float
            Multiplier for IQR (default 1.5) or z-score threshold (default 3.0)
        
        Returns:
        --------
        numpy.ndarray
            Boolean array where True indicates an outlier
        """
        if method == 'iqr':
            q1 = np.percentile(residuals, 25)
            q3 = np.percentile(residuals, 75)
            iqr = q3 - q1
            lower_bound = q1 - multiplier * iqr
            upper_bound = q3 + multiplier * iqr
            return (residuals < lower_bound) | (residuals > upper_bound)
        elif method == 'zscore':
            mean_residual = np.mean(residuals)
            std_residual = np.std(residuals)
            if std_residual == 0:
                return np.zeros_like(residuals, dtype=bool)
            z_scores = np.abs((residuals - mean_residual) / std_residual)
            return z_scores > multiplier
        else:
            raise ValueError(f"Unknown method: {method}. Use 'iqr' or 'zscore'")

    def _fit_linear_models(self, outlier_method='iqr', outlier_multiplier=1.5, min_points_after_outlier_removal=2):
        """
        Fit linear models (threshold = slope * seconds_left + intercept) for each bid value.
        Outliers are detected and removed before final fitting.
        
        Parameters:
        -----------
        outlier_method : str
            Method for outlier detection: 'iqr' (Interquartile Range) or 'zscore'
        outlier_multiplier : float
            Multiplier for outlier detection (1.5 for IQR, 3.0 for z-score)
        min_points_after_outlier_removal : int
            Minimum number of points required after outlier removal to fit a model
        """
        # Group thresholds by bid value
        bid_data = {}
        for key, value in self.delta_thresholds.items():
            bid, seconds_left = key
            bid_rounded = round(bid, 2)
            
            if bid_rounded not in bid_data:
                bid_data[bid_rounded] = {
                    'seconds_left': [],
                    'thresholds': []
                }
            
            bid_data[bid_rounded]['seconds_left'].append(seconds_left)
            bid_data[bid_rounded]['thresholds'].append(value['threshold'])
        
        # Fit linear model for each bid with outlier removal
        for bid, data in bid_data.items():
            if len(data['seconds_left']) < 2:
                # Need at least 2 points to fit a line
                continue
            
            seconds_left_arr = np.array(data['seconds_left'])
            thresholds_arr = np.array(data['thresholds'])
            
            # Initial fit to detect outliers
            coeffs_initial = np.polyfit(seconds_left_arr, thresholds_arr, 1)
            slope_initial, intercept_initial = coeffs_initial[0], coeffs_initial[1]
            
            # Calculate residuals
            predicted_initial = slope_initial * seconds_left_arr + intercept_initial
            residuals = thresholds_arr - predicted_initial
            
            # Detect outliers
            outlier_mask = self._detect_outliers(residuals, method=outlier_method, multiplier=outlier_multiplier)
            n_outliers = np.sum(outlier_mask)
            
            # Remove outliers
            seconds_left_clean = seconds_left_arr[~outlier_mask]
            thresholds_clean = thresholds_arr[~outlier_mask]
            
            # Check if we have enough points after outlier removal
            if len(seconds_left_clean) < min_points_after_outlier_removal:
                # Not enough points after outlier removal, use original data
                seconds_left_clean = seconds_left_arr
                thresholds_clean = thresholds_arr
                n_outliers = 0
                outlier_mask = np.zeros_like(seconds_left_arr, dtype=bool)
            
            # Fit final linear model with cleaned data
            coeffs = np.polyfit(seconds_left_clean, thresholds_clean, 1)
            slope, intercept = coeffs[0], coeffs[1]
            
            # Calculate R-squared for model quality
            predicted = slope * seconds_left_clean + intercept
            ss_res = np.sum((thresholds_clean - predicted) ** 2)
            ss_tot = np.sum((thresholds_clean - np.mean(thresholds_clean)) ** 2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
            
            # Also calculate R-squared on original data for comparison
            predicted_original = slope * seconds_left_arr + intercept
            ss_res_original = np.sum((thresholds_arr - predicted_original) ** 2)
            ss_tot_original = np.sum((thresholds_arr - np.mean(thresholds_arr)) ** 2)
            r_squared_original = 1 - (ss_res_original / ss_tot_original) if ss_tot_original > 0 else 0
            
            self.linear_models[bid] = {
                'slope': slope,
                'intercept': intercept,
                'n_points': len(seconds_left_clean),
                'n_points_original': len(seconds_left_arr),
                'n_outliers_removed': n_outliers,
                'r_squared': r_squared,
                'r_squared_original': r_squared_original
            }
        
        print(f"\nFitted linear models for {len(self.linear_models)} bid values (with outlier removal)")
        print("Sample linear models (first 10):")
        for i, (bid, model) in enumerate(list(self.linear_models.items())[:10]):
            print(f"  Bid {bid}: threshold = {model['slope']:.6f} * seconds_left + {model['intercept']:.6f}, "
                  f"R² = {model['r_squared']:.4f} (original: {model['r_squared_original']:.4f}), "
                  f"n_points = {model['n_points']} (removed {model['n_outliers_removed']} outliers from {model['n_points_original']})")

    def should_buy(self, delta: float, bid: float, seconds_left: float) -> bool:
        """
        Determine if action should be 'buy' based on whether delta is in top 10% 
        for the given (bid, seconds_left) combination.
        
        Returns True if delta >= 90th percentile threshold for (bid, seconds_left), False otherwise.
        Uses linear model if available, otherwise falls back to discrete lookup.
        """
        if len(self.delta_thresholds) == 0:
            return False
        
        threshold = self.get_threshold(bid, seconds_left)
        
        if threshold is None:
            # If no data for this combination, return False (conservative)
            return False
        
        return delta >= threshold

    def get_threshold(self, bid: float, seconds_left: float) -> float:
        """
        Get the 90th percentile delta threshold for a given (bid, seconds_left) combination.
        Uses linear model if available, otherwise falls back to discrete lookup.
        """
        bid_rounded = round(bid, 2)
        
        # Try to use linear model first
        if bid_rounded in self.linear_models:
            model = self.linear_models[bid_rounded]
            threshold = model['slope'] * seconds_left + model['intercept']
            return threshold
        
        # Fall back to discrete lookup
        key = (bid_rounded, int(seconds_left))
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
            
            should_buy_flag = self.should_buy(delta, bid + SPREAD, seconds_left)
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
        
        should_buy_flag = self.should_buy(delta, bid + SPREAD, seconds_left)
        threshold = self.get_threshold(bid, seconds_left)
        
        probability = 1.0 if should_buy_flag else bid
        threshold_str = f"{threshold:.2f}" if threshold is not None else "N/A"
        print(f"price: {price}, delta: {delta:.2f}, seconds_left: {seconds_left}, bid: {bid}, ask: {ask}, "
              f"threshold: {threshold_str}, should_buy: {should_buy_flag}, probability: {probability}")
        
        return probability

    def plot_false_positives(self, X, y, bids=None):
        """
        Plot false positives (predicted buy but actual label is 0).
        
        Parameters:
        -----------
        X : pandas.DataFrame
            Input features with columns ['delta', 'seconds_left', 'bid']
        y : pandas.Series or numpy.ndarray
            True labels (1 = went up, 0 = didn't go up)
        bids : list of float, optional
            List of bid values to filter/group by. Defaults to [0.1, 0.3, 0.5, 0.7, 0.9]
        """
        if len(self.delta_thresholds) == 0:
            print("Model has not been trained yet. Call fit() first.")
            return
        
        # Convert to DataFrame if needed
        if isinstance(X, pd.DataFrame):
            df = X[['delta', 'seconds_left', 'bid']].copy()
        else:
            df = pd.DataFrame(X, columns=['delta', 'seconds_left', 'bid'])
        
        if isinstance(y, pd.Series):
            df['label'] = y.values
        else:
            df['label'] = y
        
        # Make predictions
        predictions = []
        for idx, row in df.iterrows():
            delta = row['delta']
            bid = row['bid']
            seconds_left = row['seconds_left']
            should_buy = self.should_buy(delta, bid + SPREAD, seconds_left)
            predictions.append(should_buy)
        
        df['predicted_buy'] = predictions
        
        # Identify false positives: predicted buy=True, actual label=0
        false_positives = df[(df['predicted_buy'] == True) & (df['label'] == 0)].copy()
        
        print(f"\nFalse Positives Analysis:")
        print(f"  Total samples: {len(df)}")
        print(f"  False Positives: {len(false_positives)} ({100*len(false_positives)/len(df):.2f}%)")
        print(f"  True Positives: {len(df[(df['predicted_buy'] == True) & (df['label'] == 1)])}")
        print(f"  True Negatives: {len(df[(df['predicted_buy'] == False) & (df['label'] == 0)])}")
        print(f"  False Negatives: {len(df[(df['predicted_buy'] == False) & (df['label'] == 1)])}")
        
        if len(false_positives) == 0:
            print("No false positives found to plot.")
            return
        
        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Plot 1: False positives by seconds_left
        axes[0, 0].hist(false_positives['seconds_left'], bins=50, alpha=0.7, color='red', edgecolor='black')
        axes[0, 0].set_xlabel('seconds_left', fontsize=12)
        axes[0, 0].set_ylabel('Count', fontsize=12)
        axes[0, 0].set_title(f'False Positives Distribution by seconds_left\n(n={len(false_positives)})', fontsize=12)
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: False positives by bid
        if bids is None:
            bids = [0.1, 0.3, 0.5, 0.7, 0.9]
        
        bid_counts = []
        bid_labels = []
        for bid in bids:
            bid_rounded = round(bid, 2)
            count = len(false_positives[abs(false_positives['bid'] - bid_rounded) < 0.01])
            bid_counts.append(count)
            bid_labels.append(f'{bid}')
        
        axes[0, 1].bar(bid_labels, bid_counts, alpha=0.7, color='red', edgecolor='black')
        axes[0, 1].set_xlabel('Bid Value', fontsize=12)
        axes[0, 1].set_ylabel('Count', fontsize=12)
        axes[0, 1].set_title(f'False Positives Distribution by Bid\n(n={len(false_positives)})', fontsize=12)
        axes[0, 1].grid(True, alpha=0.3, axis='y')
        
        # Plot 3: False positives delta vs seconds_left (scatter)
        scatter = axes[1, 0].scatter(false_positives['seconds_left'], false_positives['delta'], 
                                    c=false_positives['bid'], cmap='viridis', 
                                    alpha=0.6, s=50, edgecolors='black', linewidths=0.5)
        axes[1, 0].set_xlabel('seconds_left', fontsize=12)
        axes[1, 0].set_ylabel('delta', fontsize=12)
        axes[1, 0].set_title(f'False Positives: delta vs seconds_left\n(n={len(false_positives)})', fontsize=12)
        axes[1, 0].grid(True, alpha=0.3)
        plt.colorbar(scatter, ax=axes[1, 0], label='bid')
        
        # Plot 4: False positives delta vs threshold
        # Calculate threshold for each false positive
        thresholds = []
        for idx, row in false_positives.iterrows():
            threshold = self.get_threshold(row['bid'], row['seconds_left'])
            thresholds.append(threshold if threshold is not None else np.nan)
        
        false_positives['threshold'] = thresholds
        false_positives_clean = false_positives.dropna(subset=['threshold'])
        
        if len(false_positives_clean) > 0:
            scatter2 = axes[1, 1].scatter(false_positives_clean['threshold'], false_positives_clean['delta'], 
                             c=false_positives_clean['bid'], cmap='viridis', 
                             alpha=0.6, s=50, edgecolors='black', linewidths=0.5)
            # Add diagonal line (delta = threshold)
            min_val = min(false_positives_clean['threshold'].min(), false_positives_clean['delta'].min())
            max_val = max(false_positives_clean['threshold'].max(), false_positives_clean['delta'].max())
            axes[1, 1].plot([min_val, max_val], [min_val, max_val], 'r--', 
                          label='delta = threshold', linewidth=2, alpha=0.7)
            axes[1, 1].set_xlabel('threshold', fontsize=12)
            axes[1, 1].set_ylabel('delta', fontsize=12)
            axes[1, 1].set_title(f'False Positives: delta vs threshold\n(n={len(false_positives_clean)})', fontsize=12)
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
            plt.colorbar(scatter2, ax=axes[1, 1], label='bid')
        else:
            axes[1, 1].text(0.5, 0.5, 'No threshold data available', 
                           ha='center', va='center', transform=axes[1, 1].transAxes)
            axes[1, 1].set_title('False Positives: delta vs threshold', fontsize=12)
        
        plt.tight_layout()
        plt.suptitle('False Positives Analysis', fontsize=16, y=1.0)
        plt.show()

    def plot_thresholds(self, bids=None, show_linear_fit=True, show_outliers=False, X=None, y=None):
        """
        Plot delta thresholds vs seconds_left for specified bid values.
        Optionally shows the fitted linear models, outliers that were removed, and false positives.
        
        Parameters:
        -----------
        bids : list of float, optional
            List of bid values to plot. Defaults to [0.1, 0.3, 0.5, 0.7, 0.9]
        show_linear_fit : bool, optional
            If True, overlay the fitted linear models on the plot. Defaults to True.
        show_outliers : bool, optional
            If True, mark outliers that were removed during fitting. Defaults to False.
        X : pandas.DataFrame, optional
            Test data with columns ['delta', 'seconds_left', 'bid'] for plotting false positives
        y : pandas.Series or numpy.ndarray, optional
            True labels (1 = went up, 0 = didn't go up) for plotting false positives
        """
        if len(self.delta_thresholds) == 0:
            print("Model has not been trained yet. Call fit() first.")
            return
        
        if bids is None:
            bids = [0.1, 0.3, 0.5, 0.7, 0.9]
        
        # Process false positives if X and y are provided
        false_positives_data = {}
        if X is not None and y is not None:
            # Convert to DataFrame if needed
            if isinstance(X, pd.DataFrame):
                df = X[['delta', 'seconds_left', 'bid']].copy()
            else:
                df = pd.DataFrame(X, columns=['delta', 'seconds_left', 'bid'])
            
            if isinstance(y, pd.Series):
                df['label'] = y.values
            else:
                df['label'] = y
            
            # Make predictions
            predictions = []
            for idx, row in df.iterrows():
                delta = row['delta']
                bid = row['bid']
                seconds_left = row['seconds_left']
                should_buy = self.should_buy(delta, bid + SPREAD, seconds_left)
                predictions.append(should_buy)
            
            df['predicted_buy'] = predictions
            
            # Identify false positives: predicted buy=True, actual label=0
            false_positives = df[(df['predicted_buy'] == True) & (df['label'] == 0)].copy()
            
            # Identify true positives: predicted buy=True, actual label=1
            true_positives = df[(df['predicted_buy'] == True) & (df['label'] == 1)].copy()
            
            # Group false positives by bid and store all data for color coding
            for bid in bids:
                bid_rounded = round(bid, 2)
                fp_for_bid = false_positives[abs(false_positives['bid'] - bid_rounded) < 0.01]
                if len(fp_for_bid) > 0:
                    false_positives_data[bid_rounded] = {
                        'seconds_left': fp_for_bid['seconds_left'].values,
                        'delta': fp_for_bid['delta'].values,
                        'bid': fp_for_bid['bid'].values
                    }
            
            # Group true positives by bid
            true_positives_data = {}
            for bid in bids:
                bid_rounded = round(bid, 2)
                tp_for_bid = true_positives[abs(true_positives['bid'] - bid_rounded) < 0.01]
                if len(tp_for_bid) > 0:
                    true_positives_data[bid_rounded] = {
                        'seconds_left': tp_for_bid['seconds_left'].values,
                        'delta': tp_for_bid['delta'].values,
                        'bid': tp_for_bid['bid'].values
                    }
            
            print(f"\nFalse Positives: {len(false_positives)} total")
            for bid_rounded, fp_data in false_positives_data.items():
                print(f"  Bid {bid_rounded}: {len(fp_data['seconds_left'])} false positives")
            
            print(f"\nTrue Positives: {len(true_positives)} total")
            for bid_rounded, tp_data in true_positives_data.items():
                print(f"  Bid {bid_rounded}: {len(tp_data['seconds_left'])} true positives")
        else:
            false_positives_data = {}
            true_positives_data = {}
        
        plt.figure(figsize=(12, 8))
        # Get color cycle to assign consistent colors to each bid
        prop_cycle = plt.rcParams['axes.prop_cycle']
        colors = prop_cycle.by_key()['color']
        bid_colors = {}  # Store color for each bid
        
        for bid in bids:
            # Round bid to match the keys in delta_thresholds
            bid_rounded = round(bid, 2)
            
            # Collect all (seconds_left, threshold) pairs for this bid
            seconds_left_values = []
            threshold_values = []
            
            for key, value in self.delta_thresholds.items():
                key_bid, key_seconds_left = key
                if abs(key_bid - bid_rounded) < 0.01:  # Allow small floating point differences
                    seconds_left_values.append(key_seconds_left)
                    threshold_values.append(value['threshold'])
            
            # Sort by seconds_left for proper line plotting
            if seconds_left_values:
                sorted_pairs = sorted(zip(seconds_left_values, threshold_values))
                seconds_left_sorted, threshold_sorted = zip(*sorted_pairs)
                
                # Identify outliers if linear model exists
                outliers_seconds = []
                outliers_thresholds = []
                inliers_seconds = []
                inliers_thresholds = []
                
                if show_outliers and bid_rounded in self.linear_models:
                    model = self.linear_models[bid_rounded]
                    seconds_left_arr = np.array(seconds_left_sorted)
                    thresholds_arr = np.array(threshold_sorted)
                    
                    # Fit initial model to detect outliers (same as in _fit_linear_models)
                    coeffs_initial = np.polyfit(seconds_left_arr, thresholds_arr, 1)
                    slope_initial, intercept_initial = coeffs_initial[0], coeffs_initial[1]
                    predicted_initial = slope_initial * seconds_left_arr + intercept_initial
                    residuals = thresholds_arr - predicted_initial
                    
                    # Detect outliers using same method
                    outlier_mask = self._detect_outliers(residuals, method='iqr', multiplier=1.5)
                    
                    outliers_seconds = seconds_left_arr[outlier_mask].tolist()
                    outliers_thresholds = thresholds_arr[outlier_mask].tolist()
                    inliers_seconds = seconds_left_arr[~outlier_mask].tolist()
                    inliers_thresholds = thresholds_arr[~outlier_mask].tolist()
                else:
                    inliers_seconds = list(seconds_left_sorted)
                    inliers_thresholds = list(threshold_sorted)
                
                # Get color for this bid (use modulo to cycle through colors if needed)
                bid_idx = bids.index(bid) % len(colors)
                bid_color = colors[bid_idx]
                bid_colors[bid_rounded] = bid_color
                
                # Plot inlier data points
                if inliers_seconds:
                    plt.plot(inliers_seconds, inliers_thresholds, marker='o', 
                            label=f'Bid = {bid} (data)', linewidth=2, markersize=4, 
                            alpha=0.6, color=bid_color)
                
                # Plot outliers if any
                if show_outliers and outliers_seconds:
                    plt.scatter(outliers_seconds, outliers_thresholds, marker='x', 
                              s=100, color='red', linewidths=2, alpha=0.8,
                              label=f'Bid = {bid} (outliers removed)', zorder=5)
                
                # Plot linear fit if available
                if show_linear_fit and bid_rounded in self.linear_models:
                    model = self.linear_models[bid_rounded]
                    if len(seconds_left_sorted) > 0:
                        seconds_left_range = np.linspace(min(seconds_left_sorted), max(seconds_left_sorted), 100)
                        threshold_fitted = model['slope'] * seconds_left_range + model['intercept']
                        plt.plot(seconds_left_range, threshold_fitted, '--', 
                                label=f'Bid = {bid} (linear fit, R²={model["r_squared"]:.3f})', 
                                linewidth=2, alpha=0.8, color=bid_color)
                
                # Plot false positives for this bid with same color as bid line
                if bid_rounded in false_positives_data:
                    fp_data = false_positives_data[bid_rounded]
                    # Use the same color as the bid line
                    fp_color = bid_colors.get(bid_rounded, 'red')
                    plt.scatter(fp_data['seconds_left'], fp_data['delta'], 
                              marker='X', s=100, color=fp_color, 
                              edgecolors='black', linewidths=1.0, 
                              alpha=0.7, zorder=10,
                              label=f'Bid = {bid} (false positives, n={len(fp_data["seconds_left"])})')
                
                # Plot true positives for this bid with same color as bid line
                if bid_rounded in true_positives_data:
                    tp_data = true_positives_data[bid_rounded]
                    # Use the same color as the bid line
                    tp_color = bid_colors.get(bid_rounded, 'green')
                    plt.scatter(tp_data['seconds_left'], tp_data['delta'], 
                              marker='o', s=80, color=tp_color, 
                              edgecolors='black', linewidths=0.8, 
                              alpha=0.6, zorder=9,
                              label=f'Bid = {bid} (true positives, n={len(tp_data["seconds_left"])})')
        
        plt.xlabel('seconds_left', fontsize=12)
        plt.ylabel('delta_threshold', fontsize=12)
        title = 'Delta Thresholds vs Seconds Left for Different Bid Values'
        if X is not None and y is not None:
            title += ' (with True/False Positives)'
        plt.title(title, fontsize=14)
        plt.legend(loc='best', fontsize=9)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

    def plot_win_rate_by_bid_seconds_left(self, X, y, spread=0.05, bids=None, seconds_left_bins=None):
        """
        Plot win rate heatmap by bid and seconds_left.
        
        Parameters:
        -----------
        X : pandas.DataFrame
            Test data with columns ['delta', 'seconds_left', 'bid']
        y : pandas.Series or numpy.ndarray
            True labels (1 = went up, 0 = didn't go up)
        spread : float, optional
            Spread to subtract from probability for trade decision. Defaults to 0.05
        bids : list of float, optional
            List of bid values to analyze. If None, uses all unique bids in data.
        seconds_left_bins : int or list, optional
            Number of bins for seconds_left or list of bin edges. If None, uses 20 bins.
        """
        # Convert to DataFrame if needed
        if isinstance(X, pd.DataFrame):
            df = X[['delta', 'seconds_left', 'bid']].copy()
        else:
            df = pd.DataFrame(X, columns=['delta', 'seconds_left', 'bid'])
        
        if isinstance(y, pd.Series):
            df['label'] = y.values
        else:
            df['label'] = y
        
        # Make predictions
        prob = self.predict_proba(df[['delta', 'seconds_left', 'bid']])
        df['probability'] = prob[:, 1]
        
        # Determine trades: action = buy if probability - spread >= bid
        df['action'] = (df['probability'] - spread >= df['bid'])
        
        # Filter to only executed trades
        trade_df = df[df['action']].copy()
        
        if len(trade_df) == 0:
            print("\nNo trades executed. Cannot plot win rate.")
            return
        
        # Create bins for seconds_left
        if seconds_left_bins is None:
            # Use 20 bins by default
            seconds_left_bins = 20
        
        if isinstance(seconds_left_bins, int):
            # Create equal-width bins
            min_seconds = trade_df['seconds_left'].min()
            max_seconds = trade_df['seconds_left'].max()
            bin_edges = np.linspace(min_seconds, max_seconds, seconds_left_bins + 1)
        else:
            # Use provided bin edges
            bin_edges = seconds_left_bins
        
        # Bin seconds_left
        trade_df['seconds_left_bin'] = pd.cut(trade_df['seconds_left'], bins=bin_edges, include_lowest=True)
        
        # Determine bid bins
        if bids is None:
            # Use all unique bids, rounded to 2 decimal places
            unique_bids = sorted(trade_df['bid'].round(2).unique())
        else:
            unique_bids = sorted(bids)
        
        # Calculate win rate for each (bid, seconds_left_bin) combination
        win_rate_data = []
        
        for bid in unique_bids:
            bid_rounded = round(bid, 2)
            bid_trades = trade_df[abs(trade_df['bid'] - bid_rounded) < 0.01]
            
            for seconds_bin in trade_df['seconds_left_bin'].cat.categories:
                bin_trades = bid_trades[bid_trades['seconds_left_bin'] == seconds_bin]
                
                if len(bin_trades) > 0:
                    win_rate = bin_trades['label'].mean()  # Fraction of winning trades
                    n_trades = len(bin_trades)
                    win_rate_data.append({
                        'bid': bid_rounded,
                        'seconds_left_bin': seconds_bin,
                        'win_rate': win_rate,
                        'n_trades': n_trades
                    })
        
        if len(win_rate_data) == 0:
            print("\nNo trade data available for plotting win rate.")
            return
        
        # Create pivot table for heatmap
        win_rate_df = pd.DataFrame(win_rate_data)
        
        # Get bin centers for plotting
        bin_centers = [interval.mid for interval in win_rate_df['seconds_left_bin'].unique()]
        bin_centers = sorted(bin_centers)
        
        # Create pivot table
        pivot_table = win_rate_df.pivot_table(
            values='win_rate',
            index='bid',
            columns='seconds_left_bin',
            aggfunc='mean'
        )
        
        # Reorder columns by bin center
        pivot_table = pivot_table.reindex(columns=sorted(pivot_table.columns, key=lambda x: x.mid))
        
        # Fill NaN values with 0 for visualization (or use a mask)
        pivot_table_filled = pivot_table.fillna(0)
        
        # Create figure with subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
        
        # Plot 1: Win rate heatmap
        # Use masked array to handle NaN values
        pivot_values = np.ma.masked_invalid(pivot_table.values)
        im = ax1.imshow(pivot_values, aspect='auto', cmap='RdYlGn', vmin=0, vmax=1, origin='lower')
        
        # Set ticks and labels
        ax1.set_xticks(range(len(pivot_table.columns)))
        ax1.set_xticklabels([f'{col.mid:.0f}' for col in pivot_table.columns], rotation=45, ha='right')
        ax1.set_yticks(range(len(pivot_table.index)))
        ax1.set_yticklabels([f'{bid:.2f}' for bid in pivot_table.index])
        
        ax1.set_xlabel('seconds_left (bin center)', fontsize=12)
        ax1.set_ylabel('Bid', fontsize=12)
        ax1.set_title('Win Rate by Bid and Seconds Left', fontsize=14)
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax1)
        cbar.set_label('Win Rate', fontsize=12)
        
        # Add text annotations for significant values
        for i in range(len(pivot_table.index)):
            for j in range(len(pivot_table.columns)):
                value = pivot_table.iloc[i, j]
                if not pd.isna(value):
                    # Get number of trades for this cell
                    n_trades = win_rate_df[
                        (win_rate_df['bid'] == pivot_table.index[i]) &
                        (win_rate_df['seconds_left_bin'] == pivot_table.columns[j])
                    ]['n_trades'].sum()
                    
                    if n_trades >= 5:  # Only show text if enough trades
                        ax1.text(j, i, f'{value:.2f}\n(n={n_trades})',
                               ha='center', va='center', fontsize=8, color='black' if value > 0.5 else 'white')
        
        # Plot 2: Number of trades heatmap
        n_trades_pivot = win_rate_df.pivot_table(
            values='n_trades',
            index='bid',
            columns='seconds_left_bin',
            aggfunc='sum'
        )
        n_trades_pivot = n_trades_pivot.reindex(columns=sorted(n_trades_pivot.columns, key=lambda x: x.mid))
        
        # Fill NaN with 0 for number of trades
        n_trades_pivot = n_trades_pivot.fillna(0)
        
        im2 = ax2.imshow(n_trades_pivot.values, aspect='auto', cmap='Blues', origin='lower')
        
        # Set ticks and labels
        ax2.set_xticks(range(len(n_trades_pivot.columns)))
        ax2.set_xticklabels([f'{col.mid:.0f}' for col in n_trades_pivot.columns], rotation=45, ha='right')
        ax2.set_yticks(range(len(n_trades_pivot.index)))
        ax2.set_yticklabels([f'{bid:.2f}' for bid in n_trades_pivot.index])
        
        ax2.set_xlabel('seconds_left (bin center)', fontsize=12)
        ax2.set_ylabel('Bid', fontsize=12)
        ax2.set_title('Number of Trades by Bid and Seconds Left', fontsize=14)
        
        # Add colorbar
        cbar2 = plt.colorbar(im2, ax=ax2)
        cbar2.set_label('Number of Trades', fontsize=12)
        
        # Add text annotations
        for i in range(len(n_trades_pivot.index)):
            for j in range(len(n_trades_pivot.columns)):
                value = n_trades_pivot.iloc[i, j]
                if not pd.isna(value) and value > 0:
                    max_val = n_trades_pivot.values[n_trades_pivot.values > 0].max() if len(n_trades_pivot.values[n_trades_pivot.values > 0]) > 0 else 1
                    ax2.text(j, i, f'{int(value)}',
                           ha='center', va='center', fontsize=9, color='white' if value > max_val * 0.5 else 'black')
        
        plt.tight_layout()
        plt.show()
        
        # Print summary statistics
        print(f"\n{'='*80}")
        print("Win Rate Summary by Bid and Seconds Left")
        print(f"{'='*80}")
        print(f"Total trades: {len(trade_df)}")
        print(f"Overall win rate: {trade_df['label'].mean():.4f}")
        print(f"Bid range: {trade_df['bid'].min():.2f} - {trade_df['bid'].max():.2f}")
        print(f"Seconds left range: {trade_df['seconds_left'].min():.0f} - {trade_df['seconds_left'].max():.0f}")
        print(f"{'='*80}\n")

    def print_pnl_by_bid(self, X, y, spread=0.05, bids=None):
        """
        Calculate and print PnL (Profit and Loss) grouped by bid values.
        
        Parameters:
        -----------
        X : pandas.DataFrame
            Test data with columns ['delta', 'seconds_left', 'bid']
        y : pandas.Series or numpy.ndarray
            True labels (1 = went up, 0 = didn't go up)
        spread : float, optional
            Spread to subtract from probability for trade decision. Defaults to 0.05
        bids : list of float, optional
            List of bid values to analyze. If None, uses all unique bids in data.
        """
        # Convert to DataFrame if needed
        if isinstance(X, pd.DataFrame):
            df = X[['delta', 'seconds_left', 'bid']].copy()
        else:
            df = pd.DataFrame(X, columns=['delta', 'seconds_left', 'bid'])
        
        if isinstance(y, pd.Series):
            df['label'] = y.values
        else:
            df['label'] = y
        
        # Make predictions
        prob = self.predict_proba(df[['delta', 'seconds_left', 'bid']])
        df['probability'] = prob[:, 1]
        
        # Determine trades: action = buy if probability - spread >= bid
        df['action'] = (df['probability'] - spread >= df['bid'])
        
        # Filter to only executed trades
        trade_df = df[df['action']].copy()
        
        # Calculate PnL
        trade_df['revenue'] = trade_df['label'].astype(float)  # is_up (1 if went up, 0 if didn't)
        trade_df['cost'] = trade_df['bid']
        trade_df['pnl'] = trade_df['revenue'] - trade_df['cost']
        
        if len(trade_df) == 0:
            print("\nNo trades executed. PnL by bid cannot be calculated.")
            return
        
        # Group by bid
        if bids is None:
            # Use all unique bids, rounded to 2 decimal places
            unique_bids = sorted(trade_df['bid'].round(2).unique())
        else:
            unique_bids = sorted(bids)
        
        print(f"\n{'='*80}")
        print("PnL by Bid")
        print(f"{'='*80}")
        print(f"{'Bid':<10} {'Trades':<10} {'Revenue':<12} {'Cost':<12} {'PnL':<12} {'Avg PnL':<12} {'Win Rate':<12}")
        print(f"{'-'*80}")
        
        total_trades = 0
        total_revenue = 0.0
        total_cost = 0.0
        total_pnl = 0.0
        
        for bid in unique_bids:
            bid_rounded = round(bid, 2)
            # Match bids within 0.01 tolerance
            bid_trades = trade_df[abs(trade_df['bid'] - bid_rounded) < 0.01]
            
            if len(bid_trades) > 0:
                n_trades = len(bid_trades)
                revenue = bid_trades['revenue'].sum()
                cost = bid_trades['cost'].sum()
                pnl = bid_trades['pnl'].sum()
                avg_pnl = bid_trades['pnl'].mean()
                win_rate = bid_trades['revenue'].mean()  # Fraction of winning trades
                
                print(f"{bid_rounded:<10.2f} {n_trades:<10} {revenue:<12.4f} {cost:<12.4f} "
                      f"{pnl:<12.4f} {avg_pnl:<12.4f} {win_rate:<12.4f}")
                
                total_trades += n_trades
                total_revenue += revenue
                total_cost += cost
                total_pnl += pnl
        
        print(f"{'-'*80}")
        print(f"{'TOTAL':<10} {total_trades:<10} {total_revenue:<12.4f} {total_cost:<12.4f} "
              f"{total_pnl:<12.4f} {total_pnl/total_trades if total_trades > 0 else 0:<12.4f} "
              f"{total_revenue/total_trades if total_trades > 0 else 0:<12.4f}")
        print(f"{'='*80}\n")

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

    # Plot thresholds
    BIDS = [0.1, 0.3, 0.5, 0.7, 0.9]

    # Make predictions
    prob = model.predict_proba(test_df[feature_cols])
    test_df['probability'] = prob[:, 1]
    ret = dataset.evaluate_model_metrics(test_df, probability_column='probability', spread=0.05)
    print(ret)
    
    # Print PnL by bid
    model.print_pnl_by_bid(test_df[feature_cols], test_df['label'], spread=0.05, bids=BIDS)
    
    # Plot win rate by bid and seconds_left
    model.plot_win_rate_by_bid_seconds_left(test_df[feature_cols], test_df['label'], spread=0.05, bids=BIDS)
    
    # Plot thresholds with false positives overlaid
    model.plot_thresholds(bids=BIDS, X=test_df[feature_cols], y=test_df['label'])

    while True:
        time.sleep(1)
 
if __name__ == "__main__":
    main()
