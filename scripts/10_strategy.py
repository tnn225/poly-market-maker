import argparse

from datetime import datetime, timedelta
import os

import pandas as pd
from poly_market_maker.dataset import Dataset

def strategy(df: pd.DataFrame):
    """
    Filter rows where:
      - delta >= -60
      - bid == 0.10
      - 60 < seconds_left < 180
    
    Returns mean of is_up for matched rows.
    """
    view = df.copy()
    view["delta"] = pd.to_numeric(view["delta"], errors="coerce")
    view["bid"] = pd.to_numeric(view["bid"], errors="coerce")
    view["seconds_left"] = pd.to_numeric(view["seconds_left"], errors="coerce")

    selected = view[
        (view["delta"] >= -60)
        & (view["bid"].round(2) == 0.10)
        & (view["seconds_left"] > 60)
        & (view["seconds_left"] < 180)
    ].copy()

    return selected

def summary_by_interval(df: pd.DataFrame, selected: pd.DataFrame):
    """
    For each interval in selected, compute min_bid, max_bid, min_delta, max_delta, is_up
    from the full df.
    """
    intervals = selected["interval"].unique()
    
    # Filter full df to only selected intervals
    interval_df = df[df["interval"].isin(intervals)].copy()
    
    summary = interval_df.groupby("interval", as_index=False).agg({
        "timestamp": "first",
        "bid": ["min", "max"],
        "delta": ["min", "max"],
        "is_up": "first"
    })
    summary.columns = ["interval", "timestamp", "min_bid", "max_bid", "min_delta", "max_delta", "is_up"]
    summary["is_up"] = summary["is_up"].astype(bool)
    summary["timestamp"] = pd.to_datetime(summary["timestamp"], unit="s").dt.strftime("%Y-%m-%d %H:%M")
    summary = summary.sort_values(["is_up", "timestamp"], ascending=[False, True])
    
    return summary

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--days", type=int, default=7)
    args = parser.parse_args()

    dataset = Dataset(days=args.days)
    df = dataset.df

    selected = strategy(df)
    
    total_rows = len(selected)
    if total_rows == 0:
        print("No rows matched the criteria")
        return
    
    is_up_mean = selected["is_up"].mean()
    is_up_count = int(selected["is_up"].sum())
    
    # Print interval summary
    summary = summary_by_interval(df, selected)
    
    total_intervals = df["interval"].nunique()
    selected_intervals = len(summary)
    interval_ratio = selected_intervals / total_intervals * 100.0 if total_intervals else 0.0
    
    is_up_count = int(summary["is_up"].sum())
    is_up_ratio = is_up_count / selected_intervals * 100.0 if selected_intervals else 0.0
    
    print(f"Selected intervals: {selected_intervals}/{total_intervals} ({interval_ratio:.1f}%)")
    print(f"is_up: {is_up_count}/{selected_intervals} ({is_up_ratio:.1f}%)")
    print(f"\nSelected intervals:")
    print(summary.to_string(index=False))

if __name__ == "__main__":
    main()
