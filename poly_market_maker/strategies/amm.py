import logging
import numpy as np
from math import sqrt
from unicodedata import bidirectional

from poly_market_maker.my_token import MyToken, Collateral
from poly_market_maker.order import Order, Side
from poly_market_maker.orderbook import OrderBook
from poly_market_maker.utils import math_round_down

SIZE = 5
MAX_IMBALANCE = 50
MAX_HEDGE_IMBALANCE = 100


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

    def set_sell_prices(self, ask: float):
        self.sell_prices = []
        for i in range(int(self.depth)):
            price = round(ask + i * self.delta, 2)
            if 0.01 <= price <= 0.99 and price >= self.up + self.spread:
                self.sell_prices.append(price)

    def set_buy_prices(self, bid: float):
        self.buy_prices = []
        for i in range(int(self.depth)):
            price = round(bid - i * self.delta, 2)
            if self.p_min <= price <= self.p_max and price <= self.up - self.spread:
                self.buy_prices.append(price)

    def set_hedge_prices(self, bid: float, up: float):
        self.hedge_prices = []
        for i in range(int(self.depth)):
            price = round(bid - i * self.delta, 2)
            if 0.01 <= price <= 0.1 and price < up:
                self.hedge_prices.append(price)

    def set_price(self, bid: float, ask: float, up: float):
        self.up = up
        self.bid = bid 
        self.ask = ask 
        self.set_sell_prices(ask)
        self.set_hedge_prices(bid, up)
        self.set_buy_prices(bid)
        
        logging.info(f"set_price bid={bid}, ask={ask}, up={up}, buy_prices={self.buy_prices} sell_prices={self.sell_prices}")

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

    def get_buy_orders(self, imbalance):
        if imbalance > MAX_IMBALANCE:
            return [] 
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

    def get_hedge_orders(self, imbalance):
        """Return buy orders with fixed capital per level"""
        if imbalance > MAX_HEDGE_IMBALANCE:
            return []
        
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

    def get_expected_orders(self, orderbook: OrderBook, bid: float, ask: float, up: float, seconds_left: int):
        orders = []
        if not bid:
            bid = 0
        if not ask:
            ask = 1
        
        open_orders = orderbook.orders
        balances = orderbook.balances

        up = round(up, 3)
        down = 1.0 - up
        imbalance = balances[MyToken.A] - balances[MyToken.B]
        percent = abs(imbalance) / (MAX_IMBALANCE * seconds_left / 900)

        self.logger.info(
            f"get_expected_orders called with params: bid={bid}, ask={ask}, up={up}, down={down} seconds_left={seconds_left}, "
            f"orderbook_orders={len(orderbook.orders)}, balances={orderbook.balances}"
        )

        self.amm_a.set_price(bid, ask, up)
        self.amm_b.set_price(round(1 - ask, 2), round(1 - bid,2), down)

        sell_orders_a = self.amm_a.get_sell_orders(balances[MyToken.A])
        sell_orders_b = self.amm_b.get_sell_orders(balances[MyToken.B])

        buy_orders_a = self.amm_a.get_buy_orders(imbalance)
        buy_orders_b = self.amm_b.get_buy_orders(-imbalance)

        hedge_orders_a = self.amm_a.get_hedge_orders(imbalance)
        hedge_orders_b = self.amm_b.get_hedge_orders(-imbalance)
        
        print(f"percent {percent}, imbalance {imbalance}")

        orders = buy_orders_a + buy_orders_b + sell_orders_a + sell_orders_b + hedge_orders_a + hedge_orders_b
    
        return orders


