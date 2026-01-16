from datetime import datetime, timedelta
import os
import pandas as pd
import numpy as np
import logging



from poly_market_maker.intervals import Interval
from scipy.stats import norm
from sklearn.metrics import (
    confusion_matrix,
    roc_auc_score,
)

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

class Dataset:
    def __init__(self, days=None):
        self.feature_cols = FEATURE_COLS
        self.days = days
        self.intervals = Interval()

        if self.days:
            self._read_dates()
            print(f" read dates rows {self.df.shape[0]}")
            self._add_target_and_is_up()
            print(f" added target and is up rows {self.df.shape[0]}")
            self._train_test_split()
            print(f" train test split rows {self.train_df.shape[0]} {self.test_df.shape[0]}")
            self.show()

    def show(self):
        print(self.train_df.head())
        print(self.test_df.head())

        print("Train/Test set overview:")
        print(f"  train_df shape: {self.train_df.shape}")
        print(f"  test_df shape:  {self.test_df.shape}")
        print(f"  train labels distribution:\n {self.train_df['label'].value_counts(normalize=True)} {self.train_df['label'].value_counts(normalize=False)}")
        print(f"  test labels distribution:\n {self.test_df['label'].value_counts(normalize=True)} {self.test_df['label'].value_counts(normalize=False)}\n")

    def _read_dates(self):
        today = datetime.now()
        dates = [(today - timedelta(days=i)).strftime("%Y-%m-%d") for i in range(self.days)]
        # dates = ["2025-12-16"]
        # Load data from CSV files for each date
        dataframes = []
        for date in dates:
            if os.path.exists(f"./data/prices/price_{date}.csv"):
                df = self._read_rows(date)
                print(f"date: {date}, df shape: {df.shape}")
                if df is not None and not df.empty:
                    dataframes.append(df)

        # Combine all dataframes
        if dataframes:
            self.df = pd.concat(dataframes, ignore_index=True)
        else:
            self.df = pd.DataFrame(columns=["timestamp", "price", "bid", "ask"])

        self.df['timestamp'] = self.df['timestamp'].astype(int)
        self.df['price'] = self.df['price'].astype(float)
        self.df['bid'] = round(self.df['bid'].astype(float), 2)
        self.df['ask'] = round(self.df['ask'].astype(float), 2) 


    def _read_rows(self, date):
        path = f"./data/prices/price_{date}.csv"
        if not os.path.exists(path):
            logger.warning(f"File not found: {path}")
            return None
        
        try:
            df = pd.read_csv(path)
            return df
        except Exception as e:
            logger.error(f"Error reading {path}: {e}")
            return None

    def _balance_df(self, df):
        grouped = df.groupby('label')['timestamp'].unique()
        #print(grouped)
        for label, timestamps in grouped.items():
            # print(label, timestamps)
            timestamps = sorted(timestamps)
            print(f"initial label {label}, len: {len(timestamps)}")

        # Find minimum count of unique timestamps across all labels
        min_timestamps_count = min(len(timestamps) for timestamps in grouped)
        # print(f"Minimum timestamp count: {min_timestamp_count}")

        for label, timestamps in grouped.items():
            # print(label, timestamps)
            timestamps = sorted(timestamps)
            if len(timestamps) > min_timestamps_count:
                timestamp = timestamps[min_timestamps_count]
                # print(f"timestamp: {timestamp}")
                df = df[(df['timestamp'] < timestamp) | (df['label'] != label)]
        #print(df.head())
        grouped = df.groupby('label')['timestamp'].unique()
        for label, timestamps in grouped.items():
            # print(label, timestamps)
            timestamps = sorted(timestamps)
            print(f"final label {label}, len: {len(timestamps)}")

        return df


    def _add_intervals(self):
        for interval in self.df['interval'].unique():
            if interval in self.price_dict and interval + 900 in self.price_dict:
                continue

            data = self.intervals.get_data('BTC', interval)
            if data:
                self.price_dict[interval] = data['openPrice']
                self.price_dict[interval + 900] = data['closePrice']

    def _add_target_and_is_up(self):
        # Initialize columns
        self.df['target'] = None
        self.df['is_up'] = None
        
        # Calculate 15-minute interval start time (rounded down to nearest 900 seconds)
        self.df['interval'] = self.df['timestamp'] // 900 * 900
        
        # Create price lookup dictionary for O(1) access
        self.price_dict = dict(zip(self.df['timestamp'], self.df['price']))
        
        # Calculate target and is_up for each unique interval
        interval_values = {}
        for interval in self.df['interval'].unique():
            target = self.price_dict.get(interval)
            
            if target is not None:
                # Get price at the end of the interval (interval + 900 seconds)
                next_interval_price = self.price_dict.get(interval + 900)
                
                if next_interval_price is not None:
                    # is_up = True if price goes up or stays same
                    interval_values[interval] = {
                        'target': target,
                        'is_up': target <= next_interval_price
                    }
        
        # Apply the same target and is_up to all rows in [interval, interval+900)
        for interval, values in interval_values.items():
            mask = (self.df['interval'] == interval)
            self.df.loc[mask, 'target'] = values['target']
            self.df.loc[mask, 'is_up'] = values['is_up']

        # Filter out rows where target or is_up couldn't be calculated
        self.df = self.df[
            (self.df['is_up'].notna())
            & (self.df['bid'].notna())
            & (self.df['ask'].notna())
            & (self.df['target'].notna())
            & ((self.df['bid'] > 0) | (self.df['ask'] > 0))
            & (self.df['price'] > 0)
            & (self.df['target'] > 0)
        ].copy()

        price_vals = self.df['price'].astype(float).to_numpy()
        target_vals = self.df['target'].astype(float).to_numpy()
        self.df["delta"] = price_vals - target_vals
        self.df["percent"] = (price_vals - target_vals) / target_vals
        self.df["log_return"] = np.log(price_vals / target_vals)


        # self.df = self.df[
        #     (self.df['delta'] > -150)
        #     & (self.df['delta'] < 150)
        # ].copy()
    
        self.df['seconds_left'] = 900 - (self.df['timestamp'] - self.df['interval'])
        self.df['time'] = self.df['seconds_left'].astype(float) / 900.
        self.df['seconds_left_log'] = np.log(self.df['seconds_left'])
                
        # Sort by timestamp to ensure proper rolling window calculation
        self.df = self.df.sort_values('timestamp').reset_index(drop=True)
        
        window = 300
        # Compute rolling per-second volatility:
        #   log_return_step = log(price_t / price_{t-1})
        #   log_return_per_sec = log_return_step / dt_seconds
        #   sigma = rolling std(log_return_per_sec) over last `window` seconds
        price_s = self.df["price"].astype(float)
        ts_s = self.df["timestamp"].astype(int)
        self.df["dt"] = ts_s.diff().fillna(1).clip(lower=1).astype(float)
        self.df["log_return_step"] = np.log(price_s / price_s.shift(1))
        self.df["log_return_per_sec"] = (self.df["log_return_step"] / self.df["dt"]).replace([np.inf, -np.inf], np.nan)
        self.df["log_return_per_sec"] = self.df["log_return_per_sec"].fillna(0.0)

        # Time-based rolling window (handles missing seconds better than sample-count rolling)
        ts_index = pd.to_datetime(self.df["timestamp"], unit="s")
        sigma_s = (
            self.df.set_index(ts_index)["log_return_per_sec"]
            .rolling(f"{int(window)}s", min_periods=2)
            .std()
        )
        self.df["sigma"] = sigma_s.to_numpy()
        self.df["sigma"] = self.df["sigma"].fillna(0.0)

        # Annualize per-second volatility: sigma_annual = sigma * sqrt(seconds_per_year)
        seconds_per_year = 365.0 * 24.0 * 60.0 * 60.0
        self.df["sigma_annual"] = self.df["sigma"] * np.sqrt(seconds_per_year)
        
        # Estimate z_score from log_return, sigma, and seconds_left
        # If sigma is per-second volatility, remaining-time volatility scales as:
        #   sigma_rem = sigma * sqrt(seconds_left)
        # So:
        #   z_score = log_return / (sigma * sqrt(seconds_left))
        time_factor = np.sqrt(self.df["seconds_left"].astype(float))
        # Avoid division by zero
        sigma_scaled = self.df['sigma'].replace(0, np.nan) * time_factor
        self.df['z_score'] = self.df['log_return'] / sigma_scaled
        # Fill NaN values (where sigma was 0) with 0
        self.df['z_score'] = self.df['z_score'].fillna(0)
        
        # Use norm.cdf to get probability estimates
        self.df["prob_est"] = norm.cdf(self.df["z_score"])

    def _train_test_split(self, test_ratio: float = 0.2):
        df_sorted = self.df.sort_values('timestamp').reset_index(drop=True)
        split_idx = int(len(df_sorted) * (1 - test_ratio))
        split_idx = max(1, min(split_idx, len(df_sorted) - 1))
        self.train_df = df_sorted.iloc[:split_idx].copy()
        self.test_df = df_sorted.iloc[split_idx:].copy()

        self.df['label'] = self.df['is_up'].astype(int)
        self.train_df['label'] = self.train_df['is_up'].astype(int)
        self.test_df['label'] = self.test_df['is_up'].astype(int)

        # print(self.train_df.head())
        # print(self.test_df.head())

        self.train_df = self._balance_df(self.train_df)
        self.test_df = self._balance_df(self.test_df)

        return self.train_df, self.test_df
 
def main():
    dataset = Dataset(days=60)
    df = dataset.df
    

    return 
if __name__ == "__main__":
    main()


