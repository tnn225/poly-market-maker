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

class OrderType:
    def __init__(self, order: Order):
        self.price = order.price
        self.side = order.side
        self.token = order.token

    def __eq__(self, other):
        if isinstance(other, OrderType):
            return (
                self.price == other.price
                and self.side == other.side
                and self.token == other.token
            )
        return False

    def __hash__(self):
        return hash((self.price, self.side, self.token))

    def __repr__(self):
        return f"OrderType[price={self.price}, side={self.side}, token={self.token}]"


class SequenceStrategy(BaseStrategy):
    def __init__(self, interval: int, shares: int, is_up: bool):
        super().__init__(interval)
        self.shares = shares
        self.is_up = is_up
        self.mytoken = MyToken.A if is_up else MyToken.B

        self.last_orders_time = 0
        self.orders = self.get_orders()
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

        self.orders = self.get_orders()

        orders = []

        buy_price = round(bid, 2)
        sell_price = round(1 - ask, 2)
        size = self.shares - self.balances[self.mytoken]
        if 0.01 <= buy_price <= 0.99 and 0.01 <= sell_price <= 0.99:
            buy_order = Order(price=buy_price, size=size, side=Side.BUY, token=MyToken.A)
            sell_order = Order(price=sell_price, size=size, side=Side.BUY, token=MyToken.B)
            if self.is_up:
                orders.append(buy_order)
            else:
                orders.append(sell_order)

        self.orders = self.get_orders()
        (orders_to_cancel, orders_to_place) = self.get_orders_to_cancel_and_place(orders)
        print(f"  Orders_to_cancel: {orders_to_cancel} Orders_to_place: {orders_to_place} Orders: {self.orders}")

        if not DEBUG and len(orders_to_cancel) + len(orders_to_place) > 0:
            self.cancel_orders(orders_to_cancel)
            self.place_orders(orders_to_place)
            self.last_orders_time = 0
            self.orders = self.get_orders()

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
            time.sleep(9)


        self.order_book_engine.stop()

        if self.balances[self.mytoken] + MIN_SIZE <= self.shares:
            self.logger.error(f"Shares is too low: {self.balances[self.mytoken] + MIN_SIZE:.2f}")
            return False

        self.logger.info("Sequence strategy finished running")
        return True

