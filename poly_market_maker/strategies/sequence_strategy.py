from poly_market_maker.orderbook import OrderBook
from poly_market_maker.constants import MIN_SIZE
from poly_market_maker.order import Order

from poly_market_maker.clob_api import ClobApi
from poly_market_maker.order_book_engine import OrderBookEngine
from poly_market_maker.my_token import MyToken
from poly_market_maker.order import Side
from poly_market_maker.price_engine import PriceEngine
import logging
import time
import math
from typing import Tuple
from poly_market_maker.strategies.base_strategy import BaseStrategy

DEBUG = False
MIN_SHARES = 10
MAX_SHARES = MIN_SHARES * 16

class SequenceStrategy(BaseStrategy):
    def __init__(self, interval: int, shares: int, is_up: bool):
        super().__init__(interval)
        self.shares = shares
        self.is_up = is_up
        self.mytoken = MyToken.A if is_up else MyToken.B

        self.last_orders_time = 0
        self.orders = self.get_orders()
        self.order = self.orders[0] if len(self.orders) > 0 else None
        self.balances = {MyToken.A: 0, MyToken.B: 0}

    def trade(self):
        self.balances = self.clob_api.get_shares(self.market)

        if self.balances[MyToken.A] + self.balances[MyToken.B] > MAX_SHARES:
            self.logger.error(f"Shares is too high: {self.balances[MyToken.A] + self.balances[MyToken.B]:.2f}")
            return

        bid, ask = self.order_book_engine.get_bid_ask(MyToken.A)

        if bid is None or ask is None:
            self.logger.error(f"No bid or ask {bid} {ask}")
            return

        delta = self.price - self.target
        self.logger.info(f"price: {self.price:.2f} ({delta:+.2f}) bid: {bid:.2f} ask: {ask:.2f}")

        price = round(bid, 2) if self.is_up else round(1 - ask, 2)
        self.order = self.get_order(self.order.id) if self.order is not None else None

        size = self.shares - self.balances[self.mytoken]
        if self.order is not None:
            size = min(size, self.order.size - self.order.size_matched)

        if 0.01 <= price <= 0.99 and size >= MIN_SIZE and (self.order is None or price > self.order.price):
            if self.order is not None:
                print(f"Cancelling order {self.order}")
                self.clob_api.cancel_order(self.order.id)
                self.order = self.get_order(self.order.id)
                size = min(size, self.order.size - self.order.size_matched)

            self.order = self.place_order(Order(price=price, size=size, side=Side.BUY, token=self.mytoken))
            print(f"Placing order {self.order}")

    def run(self):
        while int(time.time()) <= self.interval + 900 and self.balances[self.mytoken] + MIN_SIZE <= self.shares:
            time.sleep(1)

            data = self.price_engine.get_data()
            if data is None:
                self.logger.error(f"No price data {data}")
                continue

            self.price = data.get('price')
            self.target = data.get('target')
            if self.price is None or self.target is None:
                self.logger.error(f"No price {self.price} or target {self.target}")
                continue 
            self.trade()

        self.order_book_engine.stop()

        if self.balances[self.mytoken] + MIN_SIZE <= self.shares:
            self.logger.error(f"Shares is too low: {self.balances[self.mytoken] + MIN_SIZE:.2f}")
            return False

        self.logger.info("Sequence strategy finished running")
        return True

