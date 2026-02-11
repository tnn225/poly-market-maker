import csv
from datetime import datetime, timezone, timedelta
import logging
import os  
import time

import pandas as pd

from poly_market_maker.order import Order
from poly_market_maker.market import Market
from poly_market_maker.order_book_engine import OrderBookEngine
from poly_market_maker.price_engine import PriceEngine
from poly_market_maker.price_feed import PriceFeedClob
from poly_market_maker.my_token import MyToken
from poly_market_maker.utils import setup_logging
from poly_market_maker.clob_api import ClobApi
from poly_market_maker.order import Side
from poly_market_maker.strategies.simple_order import SimpleOrder
from poly_market_maker.strategies.sequence_strategy import SequenceStrategy
from poly_market_maker.binance import Binance
from poly_market_maker.utils.telegram import Telegram

from dotenv import load_dotenv          # Environment variable management
load_dotenv()                           # Load environment variables from .env file

FUNDER = os.getenv("FUNDER")
TARGET = os.getenv("TARGET")

DEBUG = False
MIN_SIZE = 5

setup_logging()
logger = logging.getLogger(__name__)

clob_api = ClobApi()
price_engine = PriceEngine(symbol="btc/usd")

telegram = Telegram()
MIN_SHARES = 20
MAX_SHARES = MIN_SHARES * 16

def get_df():
    binance = Binance(symbol="BTCUSDT", interval="15m")
    df = binance.get_df(start_time=datetime.now(timezone.utc) - timedelta(hours=24), end_time=datetime.now(timezone.utc))
    df = binance.add_features(df)
    return df 

def is_spike(df: pd.DataFrame, interval: int):
    df = df[df['open_time'] == interval]
    if df.empty:
        return False
    return df['is_spike'].iloc[0]

def get_delta(df: pd.DataFrame, interval: int):
    df = df[df['open_time'] == interval]
    if df.empty:
        return 0
    return df['delta'].iloc[0]

def get_is_up(df: pd.DataFrame, interval: int):
    df = df[df['open_time'] == interval]
    if df.empty:
        return False
    return df['is_up'].iloc[0]

def test_spike():
    df = get_df()
    df['next_is_up'] = df['is_up'].shift(-1)

    df = df[df['is_spike'] == True]
    print(df[['open_time', 'is_spike', 'delta', 'is_up', 'next_is_up']])

def run_sequence(interval: int, shares: int, is_up: bool):
    side = "Up" if is_up else "Down"
    slug = f"btc-updown-15m-{interval}"
    url = f"<a href=\"https://polymarket.com/event/{slug}\">{slug}</a>"

    if shares > MAX_SHARES: 
        telegram.send_message(f"{url} {shares} shares {side} too high")
        return False
    telegram.send_message(f"{url} {shares} shares {side}")

    strategy = SequenceStrategy(interval, shares, is_up)
    if not strategy.run():
        telegram.send_message(f"{url} {shares} shares {side} balances {strategy.balances[strategy.mytoken]}: too low")
        return False
    position = strategy.clob_api.get_position(strategy.market, strategy.market.token_ids[strategy.mytoken])
    if position is not None:
        telegram.send_message(f"{url} {position['size']:.2f} shares {side} at {position['avg_price']:.2f} = ${position['cost']:.2f}: success")
    else:
        telegram.send_message(f"{url} {shares} shares {side}: success")

    while int(time.time()) <= interval + 840:
        time.sleep(1)
        now = int(time.time())
        seconds_left = 900 - (now % 900)
        if seconds_left < 120:
            df = get_df()
            delta = get_delta(df, interval)
            if get_is_up(df, interval) != is_up and abs(delta) > 500:
                break
    df = get_df()
    if get_is_up(df, interval) != is_up:
        return run_sequence(interval+900, shares * 2, is_up)

    return True

def main():
    while True:
        time.sleep(1)
        now = int(time.time())
        seconds_left = 900 - (now % 900)

        if 60 <= seconds_left <= 840:
            print(f"seconds left: {seconds_left}")
            continue

        interval = (now + 60) // 900 * 900 
        print(f"interval: {interval}")
        previous_interval = interval - 900

        df = get_df()
        if True or is_spike(df, previous_interval):
            is_up = get_is_up(df, previous_interval)
            target = not is_up
            run_sequence(interval, MIN_SHARES, target)

if __name__ == "__main__":
    main()
