import argparse

from datetime import datetime, timedelta
import os

import pandas as pd
from poly_market_maker.dataset import Dataset
from poly_market_maker.holders import Holders
from poly_market_maker.clob_api import ClobApi

def print_intervals(df: pd.DataFrame, intervals: list):
    """Print interval, min_delta, max_delta, min_bid, max_bid, is_up for each interval."""
    view = df[
        df["interval"].isin(intervals)
        & (df["seconds_left"] >= 0)
        & (df["seconds_left"] <= 60)
    ].copy()
    if view.empty:
        print("interval,min_delta,max_delta,min_bid,max_bid,is_up")
        return
    view["delta"] = pd.to_numeric(view["delta"], errors="coerce")
    view["bid"] = pd.to_numeric(view["bid"], errors="coerce")
    agg = view.groupby("interval").agg(
        min_delta=("delta", "min"),
        max_delta=("delta", "max"),
        min_bid=("bid", "min"),
        max_bid=("bid", "max"),
        is_up=("is_up", "first"),
    ).reset_index()
    print("interval,min_delta,max_delta,min_bid,max_bid,is_up")
    for _, row in agg.iterrows():
        print(f"{row['interval']},{row['min_delta']},{row['max_delta']},{row['min_bid']},{row['max_bid']},{row['is_up']}")

def find_intervals(df: pd.DataFrame) -> list:
    """
    Return list of intervals where, in the last 60 seconds_left,
    delta changed from negative to positive and is_up is True.
    """
    required = {"interval", "seconds_left", "delta", "is_up"}
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"df missing columns for find_intervals: {missing}")

    view = df[(df["seconds_left"] >= 0) & (df["seconds_left"] <= 60) & ((df["is_up"] & (df["delta"] < 0)) | (~df["is_up"] & (df["delta"] >= 0)))].copy()
    return view["interval"].unique().tolist()

def print_top_wallets(df: pd.DataFrame, top: int = 100):
    clob_api = ClobApi()
    """Print top proxyWallet by sum of amount."""
    if df.empty or "proxyWallet" not in df.columns or "amount" not in df.columns:
        return
    agg = df.groupby("proxyWallet").agg(
        amount_sum=("amount", "sum"),
        unique_intervals=("interval", "nunique"),
    ).sort_values("amount_sum", ascending=False)
    print("proxyWallet,amount_sum,unique_intervals,trades")
    for wallet, row in agg.head(top).iterrows():
        trades = clob_api.get_traded_count(wallet)
        # if trades > 1000:
        #    continue
        print(f"{wallet},{row['amount_sum']:.2f},{row['unique_intervals']},{trades}")

def main():
    days = 2
    dataset = Dataset(days=days)
    df = dataset.df
    intervals = find_intervals(df)
    print_intervals(df, intervals)

    holders = Holders(days=days)
    df = holders.df
    df = df[df["interval"].isin(intervals)]
    print_top_wallets(df)

if __name__ == "__main__":
    main()
