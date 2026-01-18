import logging
import pandas as pd
import matplotlib.pyplot as plt

from poly_market_maker.dataset import Dataset

SPREAD = 0.05
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s"
)

FEATURE_COLS = ['delta', 'percent', 'log_return', 'time', 'seconds_left', 'bid', 'ask']

def view_interval(df):
    grouped = df.groupby(['label', 'interval']).size()
    for (label, interval), count in grouped.items():
        print(f"label {label}, interval {interval}, count: {count}")
    print("--------------------------------")



def view_delta_binance_and_dataset(binance_df, dataset_df):
    """Match Binance and dataset by timestamp and plot delta vs delta"""
    binance_df = binance_df.sort_values('open_time').copy()
    print(f"binance_df: {len(binance_df)}")
    dataset_df = dataset_df.sort_values('timestamp').copy()
    print(f"dataset_df: {len(dataset_df)}")
    
    # Create a mapping from open_time to delta in binance_df
    binance_dict = dict(zip(binance_df['open_time'], binance_df['delta']))
    print(f"binance_dict: {len(binance_dict)}")
    
    # Match dataset_df['timestamp'] with binance_df['open_time'] and add binance_delta
    dataset_df['binance_delta'] = dataset_df['timestamp'].map(binance_dict)
    
    # Remove rows where we couldn't find a match
    dataset_df = dataset_df.dropna(subset=['delta', 'binance_delta'])
    print(f"matched rows: {len(dataset_df)}")
    
    # Find indices where binance_delta < -50 or > 50
    extreme_indices = dataset_df[(dataset_df['binance_delta'] < -50) | (dataset_df['binance_delta'] > 50)].index.tolist()
    print(f"found {len(extreme_indices)} extreme points (binance_delta < -50 or > 50)")
    
    # Only show the first extreme range (100 points around first extreme point)
    if len(extreme_indices) > 0:
        first_extreme_idx = extreme_indices[0]
        pos = dataset_df.index.get_loc(first_extreme_idx)
        start = max(0, pos - 50)
        end = min(len(dataset_df), pos + 50)
        dataset_df = dataset_df.iloc[start:end].sort_values('timestamp')
        print(f"showing {len(dataset_df)} points around first extreme point at position {pos}")
    else:
        print("no extreme points found, showing first 300 points")
        if len(dataset_df) > 300:
            dataset_df = dataset_df.head(300)
    
    # Create subplots: delta plot on top, bid plot on bottom
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    
    # Top plot: timestamp on x-axis with delta and binance_delta as two lines
    ax1.plot(dataset_df['timestamp'], dataset_df['delta'], alpha=0.7, label='Dataset Delta', linewidth=1)
    ax1.plot(dataset_df['timestamp'], dataset_df['binance_delta'], alpha=0.7, label='Binance Delta', linewidth=1)
    # Mark extreme points
    extreme_df = dataset_df[(dataset_df['binance_delta'] < -50) | (dataset_df['binance_delta'] > 50)]
    if len(extreme_df) > 0:
        ax1.scatter(extreme_df['timestamp'], extreme_df['binance_delta'], color='red', s=50, alpha=0.8, label='Extreme Points', zorder=5)
    ax1.set_ylabel('Delta')
    ax1.set_title('Dataset Delta vs Binance Delta Over Time (Around Extreme Points)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Bottom plot: bid vs timestamp
    ax2.plot(dataset_df['timestamp'], dataset_df['bid'], alpha=0.7, label='Bid', linewidth=1, color='green')
    ax2.set_xlabel('Timestamp')
    ax2.set_ylabel('Bid')
    ax2.set_title('Bid Over Time')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show(block=True)
    
    return binance_df, dataset_df

def view_price_binance_and_dataset(binance_df, dataset_df):
    """Match Binance and dataset by timestamp and plot close price vs price"""
    binance_df = binance_df.sort_values('open_time').copy()
    print(f"binance_df: {len(binance_df)}")
    dataset_df = dataset_df.sort_values('timestamp').copy()
    print(f"dataset_df: {len(dataset_df)}")
    
    # Create a mapping from open_time to close price in binance_df
    binance_dict = dict(zip(binance_df['open_time'], binance_df['close']))
    print(f"binance_dict: {len(binance_dict)}")
    # Match dataset_df['timestamp'] with binance_df['open_time'] and add binance_close
    dataset_df['binance_close'] = dataset_df['timestamp'].map(binance_dict)

    
    # Remove rows where we couldn't find a match
    dataset_df = dataset_df.dropna(subset=['price', 'binance_close'])
    print(f"matched rows: {len(dataset_df)}")
    
    # Limit to first 300 points for plotting
    if len(dataset_df) > 300:
        dataset_df = dataset_df.head(300)
        print(f"limited to first 300 points for plotting")
    
    # Create subplots: price plot on top, bid plot on bottom
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    
    # Top plot: timestamp on x-axis with price and binance_close as two lines
    ax1.plot(dataset_df['timestamp'], dataset_df['price'], alpha=0.7, label='Dataset Price', linewidth=1)
    ax1.plot(dataset_df['timestamp'], dataset_df['binance_close'], alpha=0.7, label='Binance Close', linewidth=1)
    ax1.set_ylabel('Price')
    ax1.set_title('Dataset Price vs Binance Close Price Over Time')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Bottom plot: bid vs timestamp
    ax2.plot(dataset_df['timestamp'], dataset_df['bid'], alpha=0.7, label='Bid', linewidth=1, color='green')
    ax2.set_xlabel('Timestamp')
    ax2.set_ylabel('Bid')
    ax2.set_title('Bid Over Time')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show(block=True)
    
    return binance_df, dataset_df

    
def plot_z_score_by_hour_of_the_week(df):
    df = df.sort_values('timestamp')
    df['hour'] = df['hour_of_the_week'] % 24
    plt.figure(figsize=(10, 6))
    plt.scatter(df['hour'], df['sigma'])
    plt.xlabel('Hour')
    plt.ylabel('Sigma')
    plt.title('Sigma by Hour of the Week')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show(block=True)

def main():
    dataset = Dataset(days=100)
    plot_z_score_by_hour_of_the_week(dataset.df)

if __name__ == "__main__":
    main()




def show_delta_distribution(df):
    df = df.sort_values('interval')
    plt.figure(figsize=(10, 6))
    plt.hist(df['delta'], bins=100)
    plt.xlabel('Delta')
    plt.ylabel('Frequency')
    plt.title('Distribution of Delta')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show(block=True)  # Keep chart open until manually closed

def show_previous_delta_vs_is_up(df):
    df = df.sort_values('interval')
    df['previous_delta'] = df['delta'].shift(1)
    
    # Remove rows with NaN previous_delta
    df = df.dropna(subset=['previous_delta', 'is_up'])
    
    # Create buckets for previous_delta using quantile-based bins (10 bins = 10% each)
    # df['previous_delta_bucket'] = pd.qcut(df['previous_delta'], q=20, labels=False, duplicates='drop')
    bins = [-10000, -1000, -500, 0, 500, 1000, 10000]
    df['previous_delta_bucket'] = pd.cut(df['previous_delta'], bins=bins, labels=False, include_lowest=True)


    # Calculate mean is_up for each bucket
    bucket_stats = df.groupby('previous_delta_bucket').agg({
        'previous_delta': 'mean',  # Use mean of previous_delta as bucket center
        'is_up': ['mean', 'count']  # Mean and count of is_up
    })
    bucket_stats.columns = ['previous_delta_mean', 'is_up_mean', 'count']
    bucket_stats = bucket_stats.reset_index()
    
    # Remove buckets with too few samples (optional)
    bucket_stats = bucket_stats[bucket_stats['count'] > 0]
    
    # Create subplots: one for mean is_up, one for count
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10), sharex=True)
    
    # Top plot: Dot plot of mean is_up
    ax1.scatter(bucket_stats['previous_delta_mean'], bucket_stats['is_up_mean'], 
                s=50, alpha=0.7, color='blue')
    ax1.set_ylabel('Mean is_up')
    ax1.set_title('Mean is_up by Previous Delta Buckets')
    ax1.grid(True, alpha=0.3)
    
    # Bottom plot: Bar chart of count
    ax2.bar(bucket_stats['previous_delta_mean'], bucket_stats['count'], 
            width=(bucket_stats['previous_delta_mean'].max() - bucket_stats['previous_delta_mean'].min()) / len(bucket_stats) * 0.8,
            alpha=0.7, color='green')
    ax2.set_xlabel('Previous Delta (bucket center)')
    ax2.set_ylabel('Count')
    ax2.set_title('Count by Previous Delta Buckets')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show(block=True)  # Keep chart open until manually closed

def show_delta_vs_previous_delta(df):
    df = df.sort_values('interval')
    df['previous_delta'] = df['delta'].shift(1)
    plt.figure(figsize=(10, 6))
    plt.scatter(df['delta'], df['previous_delta'])
    plt.xlabel('Delta')
    plt.ylabel('Previous Delta')
    plt.title('Delta vs Previous Delta')
    plt.show(block=True)

def view_delta_binance_and_intervals(binance_df, intervals_df):
    """Match intervals with Binance data and plot delta vs binance_delta"""
    binance_df = binance_df.sort_values('open_time').copy()
    intervals_df = intervals_df.sort_values('interval').copy()
    
    # Calculate deltas
    binance_df['delta'] = binance_df['close'] - binance_df['open']
    intervals_df['delta'] = intervals_df['closePrice'] - intervals_df['openPrice']
    
    # Create a mapping from open_time to delta in binance_df
    binance_dict = dict(zip(binance_df['open_time'], binance_df['delta']))
    
    # Match intervals_df['interval'] with binance_df['open_time'] and add binance_delta
    intervals_df['binance_delta'] = intervals_df['interval'].map(binance_dict)
    
    # Remove rows where we couldn't find a match
    intervals_df = intervals_df.dropna(subset=['delta', 'binance_delta'])
    
    # Plot delta vs binance_delta
    plt.figure(figsize=(10, 6))
    plt.scatter(intervals_df['delta'], intervals_df['binance_delta'], alpha=0.5, s=20)
    plt.xlabel('Intervals Delta')
    plt.ylabel('Binance Delta')
    plt.title('Intervals Delta vs Binance Delta')
    plt.grid(True, alpha=0.3)
    
    # Add diagonal line for reference
    min_val = min(intervals_df['delta'].min(), intervals_df['binance_delta'].min())
    max_val = max(intervals_df['delta'].max(), intervals_df['binance_delta'].max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.5, label='y=x')
    plt.legend()
    
    plt.tight_layout()
    plt.show(block=True)
    
    return binance_df, intervals_df
