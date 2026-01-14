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


def strategy(df: pd.DataFrame):
    """
    Screen rows for intervals that satisfy:
      - 60 <= seconds_left <= 300
      - bid == 0.35
      - delta < 100

    Returns:
      (selected_df, summary_dict)
    """
    required = {"interval", "timestamp", "seconds_left", "bid", "delta"}
    missing = [c for c in sorted(required) if c not in df.columns]
    if missing:
        raise ValueError(f"df missing columns for strategy: {missing}")

    view = df.copy()
    view["seconds_left"] = pd.to_numeric(view["seconds_left"], errors="coerce")
    view["bid"] = pd.to_numeric(view["bid"], errors="coerce")
    view["delta"] = pd.to_numeric(view["delta"], errors="coerce")

    selected = view[
        view["seconds_left"].between(60, 300, inclusive="both")
        & (view["bid"].round(2) == 0.35)
        & (view["delta"] > -100)
    ].copy()

    summary = {
        "count_rows": int(selected.shape[0]),
        "count_intervals": int(selected["interval"].nunique()) if not selected.empty else 0,
    }
    return selected, summary

def screen_intervals():
    parser = argparse.ArgumentParser()
    parser.add_argument("--days", type=int, default=7)
    parser.add_argument("--delta-min", type=float, default=-300)
    parser.add_argument("--delta-max", type=float, default=-0)
    parser.add_argument("--bid-threshold", type=float, default=0.49)
    parser.add_argument("--first-seconds", type=int, default=300)
    parser.add_argument("--show", type=int, default=100, help="How many selected intervals to print")
    parser.add_argument("--data-dir", type=str, default="./data/prices")
    args = parser.parse_args()

    dataset = Dataset(days=args.days)
    df = dataset.df

    selected, summary = strategy(df)
    num_intervals = int(df["interval"].nunique()) if "interval" in df.columns else 0
    matched_intervals = int(summary["count_intervals"])
    matched_pct = (matched_intervals / num_intervals * 100.0) if num_intervals else 0.0
    up_intervals = 0
    up_pct = 0.0
    if matched_intervals and "is_up" in selected.columns:
        interval_is_up = (
            selected.groupby("interval", as_index=True)["is_up"]
            .first()
            .astype(bool)
        )
        up_intervals = int(interval_is_up.sum())
        up_pct = up_intervals / matched_intervals * 100.0

    print(
        f"Matched intervals: {matched_intervals}/{num_intervals} ({matched_pct:.1f}%) | "
        f"is_up=true: {up_intervals}/{matched_intervals} ({up_pct:.1f}%)"
    )

    if selected.empty:
        return

    cols = ["interval", "timestamp", "seconds_left", "bid", "ask", "delta"]
    cols = [c for c in cols if c in selected.columns]
    selected = selected.sort_values(["interval", "timestamp"])

    interval_counts = selected.groupby("interval").size().sort_values(ascending=False)
    print(f"Top intervals by matched rows:\n{interval_counts.head(args.show).to_string()}")
    print(selected[cols].head(args.show).to_string(index=False))



def main():
    screen_intervals()

if __name__ == "__main__":
    main()
