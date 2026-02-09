from datetime import datetime, timedelta
import os
import pandas as pd
import numpy as np
import logging



from poly_market_maker.intervals import Interval
from scipy.stats import norm

SPREAD = 0.01
DAYS = 30
SECONDS_LEFT_BIN_SIZE = 15
SECONDS_LEFT_BINS = int(900 / SECONDS_LEFT_BIN_SIZE)

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s"
)


FEATURE_COLS = ['seconds_left_log', 'log_return', 'delta', 'seconds_left', 'bid', 'ask', 'z_score', 'prob_est']

class Holders:
    def __init__(self, days=None):
        self.days = days
        self._read_dates()
        print(f" read dates rows {self.df.shape[0]}")

    def _read_dates(self):
        today = datetime.now()
        dates = [(today - timedelta(days=i)).strftime("%Y-%m-%d") for i in range(self.days)]
        # dates = ["2025-12-16"]
        # Load data from CSV files for each date
        dataframes = []
        for date in dates:
            if os.path.exists(f"./data/holders/holders_btc_{date}.csv"):
                df = self._read_holders(date)
                print(f"date: {date}, df shape: {df.shape}")
                dataframes.append(df)
        if dataframes:
            self.df = pd.concat(dataframes, ignore_index=True)
        else:
            self.df = pd.DataFrame(columns=["timestamp", "interval", "side", "name", "amount", "proxyWallet"])
        self.df['timestamp'] = self.df['timestamp'].astype(int)
        self.df['interval'] = self.df['interval'].astype(int)
        self.df['amount'] = self.df['amount'].astype(float)
        self.df['proxyWallet'] = self.df['proxyWallet'].astype(str)

    def _read_holders(self, date):
        path = f"./data/holders/holders_btc_{date}.csv"
        if not os.path.exists(path):
            logger.warning(f"File not found: {path}")
            return None
        
        try:
            df = pd.read_csv(path)
            df = self.filter_df(df)
            return df
        except Exception as e:
            logger.error(f"Error reading {path}: {e}")
            return None
        
    def filter_df(self, df):
        """Keep only rows with the max timestamp for each interval."""
        if df.empty or "timestamp" not in df.columns or "interval" not in df.columns:
            return df
        max_ts = df.groupby("interval")["timestamp"].transform("max")
        return df[df["timestamp"] == max_ts].copy()

 
def main():
    holders = Holders(days=2)
    df = holders.df
    print(df.head())

if __name__ == "__main__":
    main()


