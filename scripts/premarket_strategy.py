import argparse

from datetime import datetime, timedelta
import os

import pandas as pd
from poly_market_maker.dataset import Dataset

def analyze_intervals(
    df: pd.DataFrame,
    *,
    delta_min: float | None = None,
    delta_max: float | None = None,
    bid_threshold: float = 0.30,
    first_seconds: int = 450,
) -> pd.DataFrame:
    """
    For each 15-minute interval:
      delta_prev = price_at_interval_start - price_at_prev_interval_start
      if delta_prev in [delta_min, delta_max]:
        if in first `first_seconds` seconds, bid <= bid_threshold at least once:
          count is_up (interval ends up vs start)
    """
    if df.empty:
        return pd.DataFrame()

    required_cols = {"interval", "target", "seconds_left", "bid", "is_up"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"df missing required columns: {sorted(missing)}")

    # interval-level target (start price) + is_up should be constant within interval
    interval_targets = (
        df.groupby("interval", as_index=True)["target"].first().astype(float).sort_index()
    )
    interval_is_up = df.groupby("interval", as_index=True)["is_up"].first().astype(bool).sort_index()

    interval_summary = pd.DataFrame(
        {
            "target": interval_targets,
            "prev_target": interval_targets.shift(1),
            "is_up": interval_is_up,
        }
    )
    interval_summary["delta_prev"] = interval_summary["target"] - interval_summary["prev_target"]

    # First `first_seconds` seconds of the interval correspond to seconds_left >= 900-first_seconds
    seconds_left_min = 900 - int(first_seconds)
    early = df[df["seconds_left"].astype(float) >= float(seconds_left_min)].copy()
    early["bid"] = early["bid"].astype(float)

    has_low_bid = early.groupby("interval")["bid"].min() <= float(bid_threshold)
    interval_summary["has_low_bid_early"] = has_low_bid.reindex(interval_summary.index).fillna(False)

    # Filter according to the rule (delta_prev range + early bid condition)
    mask = interval_summary["has_low_bid_early"].astype(bool)
    if delta_min is not None:
        mask &= interval_summary["delta_prev"] >= float(delta_min)
    if delta_max is not None:
        mask &= interval_summary["delta_prev"] <= float(delta_max)

    selected = interval_summary[mask].copy()
    selected["interval_dt_utc"] = pd.to_datetime(selected.index, unit="s", utc=True)

    return selected.reset_index(names="interval")


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


def screen_intervals():
    parser = argparse.ArgumentParser()
    parser.add_argument("--days", type=int, default=60)
    parser.add_argument("--delta-min", type=float, default=-1000)
    parser.add_argument("--delta-max", type=float, default=-500)
    parser.add_argument("--bid-threshold", type=float, default=0.49)
    parser.add_argument("--first-seconds", type=int, default=300)
    parser.add_argument("--show", type=int, default=100, help="How many selected intervals to print")
    parser.add_argument("--data-dir", type=str, default="./data/prices")
    args = parser.parse_args()

    dataset = Dataset(days=args.days)
    df = dataset.df
    selected = analyze_intervals(
        df,
        delta_min=args.delta_min,
        delta_max=args.delta_max,
        bid_threshold=args.bid_threshold,
        first_seconds=args.first_seconds,
    )

    if selected.empty:
        print("No intervals matched.")
        return

    up = int(selected["is_up"].sum())
    total = int(selected.shape[0])
    print(
        f"Matched intervals: {total} | is_up: {up} | up_rate: {up/total:.3f} "
        f"(delta_prev in [{args.delta_min}, {args.delta_max}], early bid<={args.bid_threshold}, first {args.first_seconds}s)"
    )
    print(selected.head(args.show).to_string(index=False))

def main():
    screen_intervals()

if __name__ == "__main__":
    main()
