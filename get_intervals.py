import time
import requests_cache
import requests
import pandas as pd
from datetime import datetime, timezone

import os
import csv

session = requests_cache.CachedSession("./data/intervals.sqlite", backend="sqlite", expire_after=86400)

def get_interval_data(symbol: str, timestamp: int):
    """Get target price from Polymarket API with DataFrame caching."""
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
            if session.cache.contains(url=url) is False:
                time.sleep(2)

            response = session.get(url)
            response.raise_for_status()
            data = response.json()
            interval_data = {
                "interval": timestamp,
                "openPrice": data.get('openPrice'),
                "closePrice": data.get('closePrice'),
                "completed": data.get('completed'),
            }
            # print(f"interval_data: {interval_data}")
            if interval_data.get('completed') is False:
                session.cache.delete(url)
            return interval_data
        except Exception as e:
            print(f"Error fetching target for timestamp {timestamp}: {e}")
            time.sleep(60)
    return None 

def date_to_timestamp(date: str) -> int:
    """Convert date string (YYYY-MM-DD) to Unix timestamp (seconds)."""
    return int(datetime.strptime(date, "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp())

def main():
    """Test function to fetch target prices."""
    now = int(time.time())
    timestamp = now // 900 * 900
    while True:
        # Convert timestamp → yyyy-mm-dd
        date = datetime.fromtimestamp(timestamp, tz=timezone.utc).strftime("%Y-%m-%d")
    
        data = get_interval_data("BTC", timestamp)
        if data:
            print(f"{date} {timestamp}, openPrice: {data.get('openPrice')}, closePrice: {data.get('closePrice')}")
        else:
            print(f"{date} {timestamp} - no data")
        timestamp -= 900

if __name__ == "__main__":
    main()
