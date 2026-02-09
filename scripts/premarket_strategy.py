import argparse

from datetime import datetime, timedelta, timezone
import os

import pandas as pd
from poly_market_maker.dataset import Dataset
from poly_market_maker.binance import Binance


def main():
    days = 30
    binance = Binance(symbol="BTCUSDT", interval="15m")
    df = binance.get_df(start_time=datetime.now(timezone.utc) - timedelta(days=days), end_time=datetime.now(timezone.utc))

    df = binance.add_features(df)
    


    dataset = Dataset(days=60)
    df = dataset.df
    result = strategy(df)
    print(result)

if __name__ == "__main__":
    main()
