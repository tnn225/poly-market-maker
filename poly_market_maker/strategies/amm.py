import logging
import time

import numpy as np
from math import sqrt
from unicodedata import bidirectional

from poly_market_maker.my_token import MyToken, Collateral
from poly_market_maker.order import Order, Side
from poly_market_maker.orderbook import OrderBook
from poly_market_maker.utils import math_round_down

from poly_market_maker.models import Model
from poly_market_maker.delta_classifier import DeltaClassifier

SIZE = 5
MAX_BALANCE = 150
MAX_IMBALANCE = 50
MAX_HEDGE_IMBALANCE = 50


class AMMConfig:
    def __init__(
        self,
        p_min: float,
        p_max: float,
        spread: float,
        delta: float,
        depth: int,
        max_collateral: float,
    ):
        assert isinstance(p_min, float)
        assert isinstance(p_max, float)
        assert isinstance(delta, float)
        assert isinstance(depth, int)
        assert isinstance(spread, float)
        assert isinstance(max_collateral, float)

        self.p_min = p_min
        self.p_max = p_max
        self.delta = delta
        self.spread = spread
        self.depth = depth
        self.max_collateral = max_collateral


class AMM:
    def __init__(self, token: MyToken, config: AMMConfig):
        self.logger = logging.getLogger(self.__class__.__name__)

        assert isinstance(token, MyToken)

        # if config.spread >= config.depth:
        #    raise Exception("Depth does not exceed spread.")
        self.token = token
        self.p_min = config.p_min
        self.p_max = config.p_max
        self.delta = config.delta
        self.spread = config.spread
        self.depth = config.depth
        self.max_collateral = config.max_collateral

    def set_buy_prices(self, bid: float):
        self.buy_prices = []
        for i in range(int(self.depth)):
            price = round(bid - i * self.delta, 2)
            if self.p_min <= price <= self.p_max and price <= self.up - self.spread:
                self.buy_prices.append(price)

    def set_sell_prices(self, ask: float):
        self.sell_prices = []
        for i in range(int(self.depth)):
            price = round(ask + i * self.delta, 2)
            if 0.01 <= price <= 0.99 and price >= self.up + self.spread:
                self.sell_prices.append(price)

    def set_hedge_prices(self, bid: float, up: float):
        self.hedge_prices = []
        for i in range(int(self.depth)):
            price = round(bid - i * self.delta, 2)
            if 0.01 <= price <= 0.01 and price <= self.up:
                self.hedge_prices.append(price)

    def set_price(self, bid: float, ask: float, up: float):
        self.up = up
        self.bid = bid 
        self.ask = ask 
        self.set_sell_prices(ask)
        self.set_hedge_prices(bid, up)
        self.set_buy_prices(bid)
        
        logging.info(f"set_price bid={bid}, ask={ask}, up={up}, buy_prices={self.buy_prices} sell_prices={self.sell_prices} hedge_prices={self.hedge_prices}")

    def get_sell_orders(self, balance):
        orders = []
        for price in self.sell_prices:
            if SIZE <= balance:
                balance -= SIZE 
                orders.append(
                    Order(
                        price=price,
                        side=Side.SELL,
                        token=self.token,
                        size=SIZE,
                    )
                )
        return orders

    def get_buy_orders(self):
        """Return buy orders with fixed capital per level"""
        orders = [
            Order(
                price=price,
                side=Side.BUY,
                token=self.token,
                # size=math_round_down(CAPITAL / bid, 2),  
                size = SIZE
            )
            for price in self.buy_prices
        ]
        return orders

    def get_hedge_orders(self):
        """Return buy orders with fixed capital per level"""
        orders = [
            Order(
                price=price,
                side=Side.BUY,
                token=self.token,
                # size=math_round_down(CAPITAL / bid, 2),  
                size = 1 / price 
            )
            for price in self.hedge_prices
        ]
        return orders

class AMMManager:
    def __init__(self, config: AMMConfig):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.amm_a = AMM(token=MyToken.A, config=config)
        self.amm_b = AMM(token=MyToken.B, config=config)
        self.max_collateral = config.max_collateral
        self.p_max = config.p_max
        self.spread = config.spread
        feature_cols = ['delta', 'percent', 'log_return', 'time', 'seconds_left', 'bid', 'ask']
        self.model = Model(f"DeltaClassifier", DeltaClassifier(), feature_cols=feature_cols)
        self._last_balance_time = 0

    def get_expected_orders(self, price: float, target: float, orderbook: OrderBook, bid: float, ask: float, seconds_left: int):
        orders = []
        bid = round(bid, 2)
        ask = round(ask, 2)

        open_orders = orderbook.orders
        balances = orderbook.balances

        # get balances every 30 seconds
        current_time = time.time()
        if (current_time - self._last_balance_time) >= 60:
            self.max_balance = balances[MyToken.A] + balances[MyToken.B] + 5
            self._last_balance_time = current_time

        up = round(self.model.model.get_up(seconds_left, price - target, bid), 2)
        # down =  round(self.model.model.get_up(seconds_left, target - price - 1e-10, round(1 - ask, 2)), 2)
        down = round(1 - up, 2)

        self.amm_a.set_price(bid, ask, up)
        self.amm_b.set_price(round(1 - ask, 2), round(1 - bid, 2), down)

        sell_orders_a = [] # self.amm_a.get_sell_orders(balances[MyToken.A])
        sell_orders_b = [] # self.amm_b.get_sell_orders(balances[MyToken.B])
        
        if balances[MyToken.A] + balances[MyToken.B] <= self.max_balance:
            buy_orders_a = self.amm_a.get_buy_orders()
            buy_orders_b = self.amm_b.get_buy_orders()  
            orders += buy_orders_a + buy_orders_b   

        return orders
