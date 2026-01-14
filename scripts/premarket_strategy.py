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
    Simple edge filter + expected PnL summary.

    Filter:
      prob_est > bid + spread

    Metrics on filtered rows:
      revenue = mean(is_up)           # treats is_up as 1/0
      cost    = mean(cost_col)        # default: 'cost' if present else 'bid'
      pnl     = revenue - cost

    Returns:
      (selected_df, summary_dict)
    """
    spread = 0.1

    required = {"prob_est", "bid", "is_up"}
    missing = [c for c in sorted(required) if c not in df.columns]
    if missing:
        raise ValueError(f"df missing columns for strategy: {missing}")

    cost_col = "cost" if "cost" in df.columns else "bid"

    view = df.copy()
    view["prob_est"] = pd.to_numeric(view["prob_est"], errors="coerce")
    view["bid"] = pd.to_numeric(view["bid"], errors="coerce")
    view[cost_col] = pd.to_numeric(view[cost_col], errors="coerce")

    # is_up should be bool or 0/1; normalize to float in [0,1]
    is_up = view["is_up"]
    if is_up.dtype != bool:
        is_up = pd.to_numeric(is_up, errors="coerce")
    view["_revenue"] = is_up.astype(float)

    selected = view[view["prob_est"] > view["bid"] + spread].copy()
    if selected.empty:
        summary = {
            "spread": spread,
            "cost_col": cost_col,
            "count": 0,
            "revenue_mean": float("nan"),
            "cost_mean": float("nan"),
            "pnl_mean": float("nan"),
            "pnl_sum": 0.0,
        }
        return selected, summary

    selected["revenue"] = selected["_revenue"]
    selected["cost"] = selected[cost_col].astype(float)
    selected["pnl"] = selected["revenue"] - selected["cost"]

    revenue_mean = float(selected["revenue"].mean())
    cost_mean = float(selected["cost"].mean())
    pnl_mean = revenue_mean - cost_mean
    pnl_sum = float(selected["pnl"].sum())

    summary = {
        "spread": spread,
        "cost_col": cost_col,
        "count": int(selected.shape[0]),
        "revenue_mean": revenue_mean,
        "cost_mean": cost_mean,
        "pnl_mean": float(pnl_mean),
        "pnl_sum": pnl_sum,
    }
    return selected, summary

def screen_intervals():
    parser = argparse.ArgumentParser()
    parser.add_argument("--days", type=int, default=60)
    parser.add_argument("--delta-min", type=float, default=-300)
    parser.add_argument("--delta-max", type=float, default=-0)
    parser.add_argument("--bid-threshold", type=float, default=0.49)
    parser.add_argument("--first-seconds", type=int, default=300)
    parser.add_argument("--show", type=int, default=100, help="How many selected intervals to print")
    parser.add_argument("--data-dir", type=str, default="./data/prices")
    args = parser.parse_args()




def main():
    dataset = Dataset(days=60)
    df = dataset.df
    result = strategy(df)
    print(result)

if __name__ == "__main__":
    main()
