import argparse

from datetime import datetime, timedelta
import os

import pandas as pd
from poly_market_maker.dataset import Dataset


def print_interval():
    dataset = Dataset(days=60)
    df = dataset.df
    interval = 1764292500

    cols = ["timestamp", "price", "bid", "ask"]
    view = df[df["interval"] == interval].copy()
    if view.empty:
        print(f"No rows found for interval={interval}")
        return
    missing = [c for c in cols if c not in view.columns]
    if missing:
        raise ValueError(f"df missing columns for print_interval: {missing}")
    view = view.sort_values("timestamp")[cols]
    print(view.to_string(index=False))

def get_interval_data(df: pd.DataFrame, interval: int):
    """
    Get min_bid, max_bid, and is_up for a given interval.
    
    Returns:
      dict with keys: interval, min_bid, max_bid, is_up
    """
    required = {"interval", "bid", "is_up"}
    missing = [c for c in sorted(required) if c not in df.columns]
    if missing:
        raise ValueError(f"df missing columns for get_interval_data: {missing}")
    
    interval_df = df[df["interval"] == interval].copy()
    if interval_df.empty:
        return None
    
    interval_df["bid"] = pd.to_numeric(interval_df["bid"], errors="coerce")
    
    data = {
        "interval": int(interval),
        "min_bid": float(interval_df["bid"].min()),
        "max_bid": float(interval_df["bid"].max()),
        "is_up": bool(interval_df["is_up"].iloc[0])  # is_up is constant per interval
    }
    
    return data

def get_selected_intervals(df: pd.DataFrame):
    """
    Find intervals where in the first 5 minutes (600 <= seconds_left <= 900),
    bid = 0.05, and then filter to only those where is_up = True.
    
    Returns:
      array of unique interval IDs
    """
    required = {"interval", "seconds_left", "bid", "is_up"}
    missing = [c for c in sorted(required) if c not in df.columns]
    if missing:
        raise ValueError(f"df missing columns for get_selected_intervals: {missing}")
    
    view = df.copy()
    view["seconds_left"] = pd.to_numeric(view["seconds_left"], errors="coerce")
    view["bid"] = pd.to_numeric(view["bid"], errors="coerce")
    
    # First 5 minutes = seconds_left between 600 and 900
    first_5min = view[view["seconds_left"].between(600, 900, inclusive="both")].copy()
    
    # Filter: bid = 0.05 (rounded to 2 decimals)
    selected = first_5min[first_5min["bid"].round(2) == 0.05].copy()
    
    if selected.empty:
        return pd.array([], dtype=int)
    
    # Get unique intervals that had bid = 0.05 in first 5 minutes
    candidate_intervals = selected["interval"].unique()
    
    # Filter to only intervals where is_up = True
    # Get is_up value for each interval (should be constant per interval)
    interval_is_up = df.groupby("interval", as_index=True)["is_up"].first().astype(bool)
    
    # Filter candidate intervals to only those with is_up = True
    matched_intervals = [
        interval for interval in candidate_intervals
        if interval in interval_is_up.index and interval_is_up[interval] == True
    ]
    
    return pd.array(matched_intervals, dtype=int)

def summary_by_interval(df: pd.DataFrame):
    """
    Group by interval and compute min_bid, max_bid, and is_up for each interval.
    
    Returns:
      DataFrame with columns: interval, min_bid, max_bid, is_up
    """
    required = {"interval", "bid", "is_up"}
    missing = [c for c in sorted(required) if c not in df.columns]
    if missing:
        raise ValueError(f"df missing columns for summary_by_interval: {missing}")
    
    view = df.copy()
    view["bid"] = pd.to_numeric(view["bid"], errors="coerce")
    
    summary = view.groupby("interval", as_index=False).agg({
        "prob_est": ["min", "max"],
        "bid": ["min", "max"],
        "delta": ["min", "max"],
        "is_up": "first"
    })
    summary.columns = ["interval", "min_prob_est", "max_prob_est", "min_bid", "max_bid", "min_delta", "max_delta", "is_up"]
    summary["is_up"] = summary["is_up"].astype(bool)
    summary = summary.sort_values("interval")
    
    return summary

def show_intervals(df: pd.DataFrame, intervals: list[int]):
    for interval in intervals:
        print(interval)

def filter_by_seconds_left(df: pd.DataFrame, min_seconds_left: int, max_seconds_left: int):
    return df[(df["seconds_left"] >= min_seconds_left) & (df["seconds_left"] <= max_seconds_left)]

def main():
    dataset = Dataset(days=7)
    df = dataset.df

    df = filter_by_seconds_left(df, 0, 300)

    summary = summary_by_interval(df)

     # Filter: min_prob_est >= 0.99 AND is_up = False
    filtered = summary[
        (summary["max_prob_est"] >= 0.9999) & (summary["is_up"] == False)
    ].copy()
    
    print(f'Intervals with (max_prob_est >= 0.99) & (is_up == False): {len(filtered)}')
    print(filtered.to_string(index=False))

if __name__ == "__main__":
    main()
