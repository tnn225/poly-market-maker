import time

import pandas as pd

from datetime import datetime, timezone, timedelta

from poly_market_maker.order import Order, Side
from poly_market_maker.my_token import MyToken
from poly_market_maker.clob_api import ClobApi
from poly_market_maker.strategies.simple_order import SimpleOrder
from poly_market_maker.binance import Binance
from poly_market_maker.utils.telegram import Telegram



telegram = Telegram()
MIN_SIZE = 10

clob_api = ClobApi()

class TradeManager:
    def __init__(self, interval: int):
        self.clob_api = clob_api
        self.market = clob_api.get_market(interval)
        self.interval = interval

    def place_order(self, new_order: Order) -> Order:
        order_id = self.clob_api.place_order(
            price=new_order.price,
            size=new_order.size,
            side=new_order.side.value,
            token_id=self.market.token_id(new_order.token),
        )
        return Order(
            price=new_order.price,
            size=new_order.size,
            side=new_order.side,
            id=order_id,
            token=new_order.token,
        )

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

def is_up(df: pd.DataFrame, interval: int):
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
    if shares > 160: 
        return False
    trade_manager = TradeManager(interval)
    order = Order(price=0.50, size=shares, side=Side.BUY, token=MyToken.A.value if is_up else MyToken.B.value)
    trade_manager.place_order(order)

    while int(time.time()) <= interval + 900:
        time.sleep(1)

    df = get_df()

    if is_up(df, interval) != is_up:
        run_sequence(interval+900, shares * 2, is_up)

def main():
    while True:
        time.sleep(1)
        now = int(time.time())
        if now % 900 != 0:
            print(f"not 900 seconds: {now % 900}")
            continue

        seconds_left = 900 - (now % 900)
        interval = now // 900 * 900 
        print(f"interval: {interval}")
        previous_interval = interval - 900

        df = get_df()
        if is_spike(df, previous_interval):
            is_up = is_up(df, previous_interval)
            target = not is_up
            run_sequence(interval, 10, target)

if __name__ == "__main__":
    main()
