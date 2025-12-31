from datetime import datetime, timezone, timedelta
import time
import pandas as pd
import json
import requests

from poly_market_maker.utils.cache import KeyValueStore

DAYS = 10

FEATURE_COLS = ['interval', 'openPrice', 'closePrice']


class Interval:
    def __init__(self, days=DAYS):
        self.days = days
        self.cache = KeyValueStore(db_path="./data/intervals.sqlite")
        self.read_dates()

    def get_data(self, symbol: str, timestamp: int, only_cache=False):
        """Get target price from Polymarket API."""
        # print(f"Getting data for {symbol} at {timestamp}")
        timestamp = timestamp // 900 * 900

        if self.cache.exists(timestamp):
            return json.loads(self.cache.get(timestamp))
        if only_cache:
            return None

        # Fetch from API
        eventStartTime = datetime.fromtimestamp(timestamp, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        endDate = datetime.fromtimestamp(timestamp + 900, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        params = {
            "symbol": symbol,
            "eventStartTime": eventStartTime,
            "variant": "fifteen",
            "endDate": endDate
        }
        
        url = requests.Request('GET', "https://polymarket.com/api/crypto/crypto-price", params=params).prepare().url
        
        for i in range(3):
            try:
                print(f"Getting data from {url}")

                response = requests.get(url)
                response.raise_for_status()
                data = response.json()
                interval_data = {
                    "interval": timestamp,
                    "openPrice": data.get('openPrice'),
                    "closePrice": data.get('closePrice'),
                    "completed": data.get('completed'),
                }
                # print(f"interval_data: {interval_data}")
                if interval_data.get('completed'):
                    # print(f"Caching data for timestamp {timestamp}")
                    self.cache.set(timestamp, json.dumps(interval_data))
                return interval_data
            except Exception as e:
                print(f"Error fetching target for timestamp {timestamp}: {e}")
                time.sleep(60)
        return None 


    def date_to_timestamp(self, date: str) -> int:
        """Convert date string (YYYY-MM-DD) to Unix timestamp (seconds)."""
        return int(datetime.strptime(date, "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp())

    def read_dates(self):
        today = datetime.now()
        dates = [(today - timedelta(days=i+1)).strftime("%Y-%m-%d") for i in range(self.days)]
        # Load data from CSV files for each date
        dataframes = []
        for date in dates:
            time.sleep(1)
            timestamp = self.date_to_timestamp(date)
            for i in range(96):
                data = self.get_data("BTC", timestamp + i * 900)
                if data is not None:
                    row = [data.get('interval'), data.get('openPrice'), data.get('closePrice')]
                    dataframes.append(row)
        self.df = pd.DataFrame(dataframes, columns=FEATURE_COLS)
        self.df['delta'] = self.df['closePrice'] - self.df['openPrice']
        self.df['is_up'] = self.df['closePrice'] >= self.df['openPrice']
        return self.df

def main():
    intervals = Interval()
    df = intervals.df
    print(df.head())

if __name__ == "__main__":
    main()