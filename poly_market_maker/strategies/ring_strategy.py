from poly_market_maker.orderbook import OrderBook
from poly_market_maker.constants import MIN_SIZE, DEBUG
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
from poly_market_maker.strategies.base_strategy import BaseStrategy, OrderType
from poly_market_maker.strategies.ring_order import RingOrder

MAX_BALANCE = 5000

class RingStrategy(BaseStrategy):
    def __init__(self, interval: int):
        super().__init__(interval)

        self.spread = 0.03  # Spread for paired orders
        self.ring_a = RingOrder(token=MyToken.A, market=self.market)
        self.ring_b = RingOrder(token=MyToken.B, market=self.market)
        self.ring_a.set_other(self.ring_b)
        self.ring_b.set_other(self.ring_a)

    def trade(self):
        self.balances = self.get_balances()
        if self.balances[MyToken.A] + self.balances[MyToken.B] > MAX_BALANCE:
            self.logger.error(f"Balance is too high: {self.balances[MyToken.A] + self.balances[MyToken.B]:.2f}")
            return
        self.logger.info(f"balances: A {self.balances[MyToken.A]:.2f} B {self.balances[MyToken.B]:.2f}")  

        bid, ask = self.order_book_engine.get_bid_ask(MyToken.A)
        if bid is None or ask is None:
            self.logger.error(f"No bid or ask")
            return

        delta = self.price - self.target
        inventory = self.balances[MyToken.A] - self.balances[MyToken.B]
 
        self.ring_a.trade(inventory, delta, bid, ask)
        self.ring_b.trade(-inventory, -delta, 1-ask, 1-bid)

    def run(self):
        while int(time.time()) <= self.end_time:
            time.sleep(1)
            self.seconds_left = self.end_time - int(time.time())

            data = self.price_engine.get_data()
            if data is None:
                self.logger.info(f"No price data")
                continue

                
            self.price = data.get('price')
            self.target = data.get('target')
            if self.price is None or self.target is None:
                self.logger.info(f"No price or target")
                return
            self.trade()

        self.order_book_engine.stop()
        self.logger.info("Ring strategy finished running")