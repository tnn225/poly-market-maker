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


class PairStrategy(BaseStrategy):
    def __init__(self, interval: int):
        self.clob_api = ClobApi()
        self.price_engine = PriceEngine(symbol="btc/usd")
        self.market = self.clob_api.get_market(interval)

        self.spread = 0.03

        self.start_time = interval
        self.end_time = interval + 900
        
        self.logger = logging.getLogger(self.__class__.__name__)
        
        self.order_book_engine = OrderBookEngine(self.market)
        self.order_book_engine.start()

        self.last_orders_time = 0
        self.orders = self.get_orders()
        self.balances = {MyToken.A: 0, MyToken.B: 0}
        self.pairs = {}
        self.order_types = set(OrderType(order) for order in self.orders)

    def get_balances(self) -> dict:
        self.balances = self.clob_api.get_balances(self.market)
        return self.balances

    def get_orders(self) -> list[Order]:
        if time.time() - self.last_orders_time < 10:
            return self.orders
        
        self.balances = self.get_balances()
        
        order_dicts = self.clob_api.get_orders(self.market.condition_id)
        self.orders = [
            Order(
                size=float(order_dict["size"]),
                price=float(order_dict["price"]),
                side=Side(order_dict["side"]),
                token=self.market.token(order_dict["token_id"]),
                id=order_dict["id"],
            )
            for order_dict in order_dicts
        ]
        self.order_types = set(OrderType(order) for order in self.orders)
        self.last_orders_time = time.time()
        return self.orders

    def is_valid_order(self, order: Order) -> bool:
        order_type = OrderType(order)
        if order_type in self.order_types:
            return False
        if order in self.pairs:
            order_type = self.pairs[order]
            if order_type in self.order_types:
                return False
        return True

    def place_orders(self, orders: list[Order]):
        for order in orders:
            self.place_order(order)

    def place_order(self, new_order: Order) -> Order:
        order_id = self.clob_api.place_order(
            price=new_order.price,
            size=new_order.size,
            side=new_order.side.value,
            token_id=self.market.token_id(new_order.token),
        )

    def get_orders_to_cancel_and_place(self, expected_orders: list[Order]) -> Tuple[list[Order], list[Order]]:
        orders_to_cancel, orders_to_place = [], []

        expected_order_types = set(OrderType(order) for order in expected_orders)

        orders_to_cancel += list(
            filter(
                lambda order: (
                    OrderType(order) not in expected_order_types and order.size == MIN_SIZE
                ),
                self.orders,
            )
        )

        for order_type in expected_order_types:
            open_orders = [
                order for order in self.orders if OrderType(order) == order_type
            ]
            open_size = sum(order.size for order in open_orders)
            expected_size = sum(
                order.size
                for order in expected_orders
                if OrderType(order) == order_type
            )

            # Cancel all existing orders of this type if:
            # 1. Total size doesn't match, OR
            # 2. There are multiple orders (duplicates) - always consolidate to a single order
            if open_size != expected_size or len(open_orders) > 1:
                orders_to_cancel += open_orders
                new_size = expected_size
            # otherwise get the remaining size
            else:
                new_size = round(expected_size - open_size, 2)

            if new_size >= MIN_SIZE:
                orders_to_place += [
                    self._new_order_from_order_type(order_type, new_size)
                ]

        return (orders_to_cancel, orders_to_place)

    def trade(self):
        self.balances = self.get_balances()
        if self.balances[MyToken.A] + self.balances[MyToken.B] > MAX_BALANCE:
            self.logger.error(f"Balance is too high: {self.balances[MyToken.A] + self.balances[MyToken.B]:.2f}")
            return

        bid, ask = self.order_book_engine.get_bid_ask(MyToken.A)

        if bid is None or ask is None:
            self.logger.error(f"No bid or ask")
            return

        delta = self.price - self.target
        inventory = self.balances[MyToken.A] - self.balances[MyToken.B] #  - delta * 2
        self.logger.info(f"bid: {bid:.4f}, ask: {ask:.4f} Inventory: {inventory:+.2f} ")

        self.orders = self.get_orders()

        orders = []

        buy_up = 1 if (delta >= 0 and inventory == 0) or inventory < 0 else 0

        buy_price = round(bid - self.spread * (1 - buy_up), 2)
        sell_price = round(1 - ask - self.spread * buy_up, 2)
        if 0.01 <= buy_price <= 0.99 and 0.01 <= sell_price <= 0.99:
            buy_order = Order(price=buy_price, size=MIN_SIZE, side=Side.BUY, token=MyToken.A)
            sell_order = Order(price=sell_price, size=MIN_SIZE, side=Side.BUY, token=MyToken.B)
            if self.is_valid_order(buy_order) and self.is_valid_order(sell_order):
                orders.append(buy_order)
                orders.append(sell_order)
                self.pairs[buy_order] = sell_order
                self.pairs[sell_order] = buy_order

        self.logger.info(f"Orders: {orders}")
        if not DEBUG and len(orders) > 0:
            self.place_orders(orders)
            self.last_orders_time = 0
            self.orders = self.get_orders()

    def run(self):
        while int(time.time()) <= self.end_time:
            time.sleep(1)

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

