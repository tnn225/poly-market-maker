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
MAX_BALANCE = 5000

class BalanceStrategy(BaseStrategy):
    def __init__(self, interval: int):
        super().__init__(interval)

        self.depth = 5
        self.delta = 0.01

    def trade(self):
        if self.balances[MyToken.A] + self.balances[MyToken.B] > MAX_BALANCE:
            self.logger.error(f"Balance is too high: {self.balances[MyToken.A] + self.balances[MyToken.B]:.2f}")
            return

        bid, ask = self.order_book_engine.get_bid_ask(MyToken.A)

        if bid is None or ask is None:
            self.logger.error(f"No bid or ask")
            return

        delta = self.price - self.target
        inventory = self.balances[MyToken.A] - self.balances[MyToken.B] # - delta * 2 
        
        self.logger.info(f"trade {self.seconds_left} seconds left price: {self.price:.4f} {delta:+.4f} bid: {bid:.2f}, ask: {ask:.2f}")
        self.logger.info(f"  inventory: {inventory:+.2f} balances: {self.balances[MyToken.A]:.2f} {self.balances[MyToken.B]:.2f}")
        self.orders = self.get_orders()

        orders = []
        for i in range(self.depth):
            buy_price = round(bid - i * self.delta, 2)
            sell_price = round(1 - ask - i * self.delta, 2)

            if 0.01 <= buy_price <= 0.99:
                buy_order = Order(price=buy_price, size=MIN_SIZE, side=Side.BUY, token=MyToken.A)
                if inventory < -MIN_SIZE:
                    inventory += MIN_SIZE                
                    orders.append(buy_order)
                    continue
            
            if 0.01 <= sell_price <= 0.99:
                sell_order = Order(price=sell_price, size=MIN_SIZE, side=Side.BUY, token=MyToken.B)
                if inventory > MIN_SIZE:
                    inventory -= MIN_SIZE                
                    orders.append(sell_order)
                    continue

            if 0.01 <= sell_price <= 0.99 and 0.01 <= buy_price <= 0.99:
                buy_order = Order(price=buy_price, size=MIN_SIZE, side=Side.BUY, token=MyToken.A)
                sell_order = Order(price=sell_price, size=MIN_SIZE, side=Side.BUY, token=MyToken.B)
                orders.append(buy_order)
                orders.append(sell_order)

        orders_to_cancel, orders_to_place = self.get_orders_to_cancel_and_place(orders) 
        self.logger.info(f"  Orders to cancel: {len(orders_to_cancel)} Orders to place: {len(orders_to_place)}")
        self.logger.info(f"  Orders to cancel: {orders_to_cancel} Orders to place: {orders_to_place}")
        if not DEBUG and len(orders_to_cancel) + len(orders_to_place) > 0:
            self.place_orders(orders_to_place)
            self.cancel_orders(orders_to_cancel)
            self.last_orders_time = 0
            self.orders = self.get_orders()

    def run(self):
        while int(time.time()) <= self.end_time:
            time.sleep(1)
            self.seconds_left = self.end_time - int(time.time())

            data = self.price_engine.get_data()
            if data is None:
                self.logger.error(f"No price data")
                continue

                
            self.price = data.get('price')
            self.target = data.get('target')
            if self.price is None or self.target is None:
                self.logger.error(f"No price or target")
                return
            self.trade()

        self.order_book_engine.stop()
        self.logger.info("Avellaneda market maker finished running")

