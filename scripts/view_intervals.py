

from poly_market_maker.intervals import Interval
import matplotlib.pyplot as plt
import pandas as pd



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
    bins = [-10000, -1000, -10, +10, 500, 1000, 10000]
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


def main():
    intervals = Interval(days=29)
    df = intervals.df
    show_previous_delta_vs_is_up(df)

if __name__ == "__main__":
    main()