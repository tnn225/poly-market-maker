from datetime import datetime, timezone, timedelta
import os
import time
import pandas as pd
import numpy as np
import logging
import sklearn.calibration
import csv

from collections import deque

from lightgbm import LGBMClassifier

from scipy.stats import norm
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False
    print("Warning: seaborn not available, using matplotlib for heatmap")
from sklearn.metrics import brier_score_loss, log_loss

from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)

from poly_market_maker.dataset import Dataset
from poly_market_maker.models import Model
from poly_market_maker.models.tensorflow_classifier import TensorflowClassifier
from poly_market_maker.models.bucket_classifier import BucketClassifier
from poly_market_maker.models.bid_classifier import BidClassifier

SPREAD = 0.05
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s"
)

FEATURE_COLS = ['delta', 'percent', 'log_return', 'time', 'seconds_left', 'bid', 'ask']

def heatmap(df, n_delta_bins=100, n_seconds_bins=30, color_by='mean_label', figsize=(14, 10)):
    """
    Create a heatmap showing label distribution across seconds_left (x-axis) and delta (y-axis).
    
    Args:
        df: DataFrame with columns 'seconds_left', 'delta', and 'label' (optionally 'interval')
        n_delta_bins: Number of bins for delta (y-axis)
        n_seconds_bins: Number of bins for seconds_left (x-axis)
        color_by: What to color by:
            - 'mean_label': Mean label value (0-1 probability) - default
            - 'count': Sample count per bin
            - 'proportion': Proportion of positive labels
            - 'unique_intervals': Count of unique intervals per bin (requires 'interval' column)
        figsize: Figure size tuple
    """
    # Filter out NaN values and include interval if available
    required_cols = ['seconds_left', 'delta', 'label']
    if 'interval' in df.columns:
        required_cols.append('interval')
    
    plot_df = df[required_cols].dropna().copy()
    
    if len(plot_df) == 0:
        print("No data available for heatmap")
        return
    
    has_interval = 'interval' in plot_df.columns
    
    # Create bins
    # Delta bins (y-axis) - use quantiles to handle skewed distribution
    delta_min, delta_max = plot_df['delta'].min(), plot_df['delta'].max()
    if delta_max - delta_min > 0:
        # Use quantile-based bins for better distribution
        delta_bins = pd.qcut(
            plot_df['delta'], 
            q=min(n_delta_bins, plot_df['delta'].nunique()), 
            duplicates='drop',
            retbins=True
        )[1]
        # Ensure we cover the full range
        delta_bins[0] = delta_min - 1e-10
        delta_bins[-1] = delta_max + 1e-10
    else:
        delta_bins = np.linspace(delta_min - 1, delta_max + 1, n_delta_bins + 1)
    
    # Seconds_left bins (x-axis) - uniform bins from 0 to 900
    seconds_min, seconds_max = plot_df['seconds_left'].min(), plot_df['seconds_left'].max()
    seconds_bins = np.linspace(seconds_min, seconds_max, n_seconds_bins + 1)
    
    # Assign bins
    plot_df['delta_bin'] = pd.cut(plot_df['delta'], bins=delta_bins, labels=False, include_lowest=True)
    plot_df['seconds_left_bin'] = pd.cut(plot_df['seconds_left'], bins=seconds_bins, labels=False, include_lowest=True)
    
    # Remove rows with NaN bins
    plot_df = plot_df.dropna(subset=['delta_bin', 'seconds_left_bin'])
    
    # Create pivot table based on color_by parameter
    if color_by == 'mean_label':
        # Mean label value (0-1, represents probability)
        pivot_data = plot_df.groupby(['delta_bin', 'seconds_left_bin'])['label'].mean().unstack(fill_value=np.nan)
        vmin, vmax = 0, 1
        cmap = 'RdYlGn'  # Red-Yellow-Green: red=0, green=1
        label_text = 'Mean Label (Probability)'
    elif color_by == 'count':
        # Sample count
        pivot_data = plot_df.groupby(['delta_bin', 'seconds_left_bin']).size().unstack(fill_value=0)
        vmin, vmax = None, None
        cmap = 'Blues'
        label_text = 'Sample Count'
    elif color_by == 'proportion':
        # Proportion of positive labels
        pivot_data = plot_df.groupby(['delta_bin', 'seconds_left_bin'])['label'].mean().unstack(fill_value=np.nan)
        vmin, vmax = 0, 1
        cmap = 'RdYlGn'
        label_text = 'Proportion of Positive Labels'
    elif color_by == 'unique_intervals':
        # Count of unique intervals per bin
        if not has_interval:
            raise ValueError("'interval' column not found in dataframe. Cannot calculate unique intervals.")
        pivot_data = plot_df.groupby(['delta_bin', 'seconds_left_bin'])['interval'].nunique().unstack(fill_value=0)
        vmin, vmax = None, None
        cmap = 'Purples'
        label_text = 'Unique Intervals Count'
    else:
        raise ValueError(f"color_by must be 'mean_label', 'count', 'proportion', or 'unique_intervals', got '{color_by}'")
    
    # Also calculate unique intervals count for statistics (if available)
    unique_intervals_data = None
    if has_interval:
        unique_intervals_data = plot_df.groupby(['delta_bin', 'seconds_left_bin'])['interval'].nunique().unstack(fill_value=0)
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Reverse axes: x-axis (900→0), y-axis (negative→positive)
    # Flip x-axis: reverse columns (seconds_left: 900→0)
    pivot_data = pivot_data.iloc[:, ::-1]
    # Flip y-axis: reverse rows (delta: negative→positive)
    pivot_data = pivot_data.iloc[::-1, :]
    
    # Also reverse unique_intervals_data if it exists
    if unique_intervals_data is not None:
        unique_intervals_data = unique_intervals_data.iloc[:, ::-1]  # Reverse x-axis
        unique_intervals_data = unique_intervals_data.iloc[::-1, :]  # Reverse y-axis
    
    # Create heatmap
    if HAS_SEABORN:
        sns.heatmap(
            pivot_data,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            cbar_kws={'label': label_text},
            ax=ax,
            xticklabels=False,
            yticklabels=False,
            rasterized=True  # For better performance with large datasets
        )
    else:
        # Fallback to matplotlib imshow
        im = ax.imshow(
            pivot_data.values,
            cmap=cmap,
            aspect='auto',
            vmin=vmin,
            vmax=vmax,
            origin='lower',
            interpolation='nearest'
        )
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label(label_text)
    
    # Set labels and title
    ax.set_xlabel('Seconds Left (900 = start of interval, 0 = end)', fontsize=12)
    ax.set_ylabel('Delta (Price - Target, negative → positive)', fontsize=12)
    ax.set_title(f'Label Distribution Heatmap\n(color by: {label_text})', fontsize=14, fontweight='bold')
    
    # Add tick labels at reasonable intervals
    n_x_ticks = 10
    n_y_ticks = 10
    
    # X-axis ticks (seconds_left) - reversed: 900→0
    x_tick_positions = np.linspace(0, len(seconds_bins) - 2, n_x_ticks).astype(int)
    # Reverse the bin labels to match reversed axis
    reversed_seconds_bins = seconds_bins[::-1]
    x_tick_labels = [f'{int(reversed_seconds_bins[i])}' for i in x_tick_positions]
    ax.set_xticks(x_tick_positions)
    ax.set_xticklabels(x_tick_labels, rotation=0)
    
    # Y-axis ticks (delta) - reversed: negative→positive
    y_tick_positions = np.linspace(0, len(delta_bins) - 2, n_y_ticks).astype(int)
    # Reverse the bin labels to match reversed axis
    reversed_delta_bins = delta_bins[::-1]
    y_tick_labels = [f'{reversed_delta_bins[i]:.0f}' for i in y_tick_positions]
    ax.set_yticks(y_tick_positions)
    ax.set_yticklabels(y_tick_labels, rotation=0)
    
    # Add grid lines for better readability
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    
    # Add statistics text
    total_samples = len(plot_df)
    positive_samples = plot_df['label'].sum()
    positive_rate = positive_samples / total_samples if total_samples > 0 else 0
    
    stats_text = f'Total samples: {total_samples:,}\nPositive rate: {positive_rate:.2%}'
    
    # Add unique intervals count if available
    if has_interval:
        total_unique_intervals = plot_df['interval'].nunique()
        stats_text += f'\nUnique intervals: {total_unique_intervals:,}'
    
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
            fontsize=10)
    
    plt.tight_layout()
    
    # Save figure
    os.makedirs('data/plots', exist_ok=True)
    filename = f'data/plots/heatmap_{color_by}.png'
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"Heatmap saved to: {filename}")
    
    plt.show()
    
    return fig, ax


def main():
    dataset = Dataset()
    train_df = dataset.train_df
    test_df = dataset.test_df

    # Combine train and test for full dataset heatmap
    # Make sure label column exists
    if 'label' not in dataset.df.columns:
        dataset.df['label'] = dataset.df['is_up'].astype(int) if 'is_up' in dataset.df.columns else 0
    
    # Create heatmap with different color schemes
    print("Creating heatmap with mean label (probability)...")
    heatmap(dataset.df, color_by='mean_label')
    
    print("\nCreating heatmap with sample count...")
    heatmap(dataset.df, color_by='count')
    
    # Create heatmap with unique intervals count (if interval column exists)
    if 'interval' in dataset.df.columns:
        print("\nCreating heatmap with unique intervals count...")
        heatmap(dataset.df, color_by='unique_intervals')
    else:
        print("\nSkipping unique intervals heatmap (interval column not found)")

if __name__ == "__main__":
    main()