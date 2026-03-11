import time
import logging
from typing import Tuple

from poly_market_maker.orderbook import OrderBook
from poly_market_maker.order import Order
from poly_market_maker.order import Side
from poly_market_maker.constants import MIN_SIZE
from poly_market_maker.clob_api import ClobApi
from poly_market_maker.price_engine import PriceEngine
from poly_market_maker.order_book_engine import OrderBookEngine

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

class BaseStrategy:
    """Base market making strategy"""

    def __init__(self, interval: int, symbol: str = "btc", duration: int = 15):
        self.logger = logging.getLogger(self.__class__.__name__)

        self.interval = interval
        self.start_time = interval
        self.end_time = interval + 900

        self.clob_api = ClobApi()
        self.price_engine = PriceEngine(symbol=f"{symbol}/usd")
        self.market = self.clob_api.get_market(interval, symbol, duration=duration)

        self.order_book_engine = OrderBookEngine(self.market)
        self.order_book_engine.start()

        # Lists of orders to act on (do not shadow method names).
        self.orders_to_place: list[Order] = []
        self.orders_to_cancel: list[Order] = []
        self.last_orders_time = 0
        self.orders = self.get_orders()

        

    def get_orders_to_cancel_and_place(self, expected_orders: list[Order]) -> list[Order]:
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
                order for order in self.orders if OrderType(order) == order_type and order.size == MIN_SIZE
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

    @staticmethod
    def _new_order_from_order_type(order_type: OrderType, size: float) -> Order:
        return Order(
            price=order_type.price,
            size=size,
            side=order_type.side,
            token=order_type.token,
        )

    def get_balances(self) -> dict:
        self.balances = self.clob_api.get_balances(self.market)
        return self.balances



    def place_orders(self, orders: list[Order]):
        for order in orders:
            self.place_order(order)
    
    def place_orders(self, orders: list[Order]):
        for order in orders:
            self.place_order(order)

    def place_order(self, new_order: Order) -> Order:
        order = self.clob_api.place_order(
            price=new_order.price,
            size=new_order.size,
            side=new_order.side.value,
            token_id=self.market.token_id(new_order.token),
        )
        return Order(
            price=new_order.price,
            size=new_order.size,
            side=new_order.side,
            id=order,
            token=new_order.token,
        )

    def cancel_orders(self, orders: list[Order]) -> list[Order]:
        order_ids = [order.id for order in orders]
        self.clob_api.cancel_orders(order_ids)

    def get_orders(self) -> list[Order]:
        if time.time() - self.last_orders_time < 3:
            return self.orders
        
        self.balances = self.get_balances()
        
        order_dicts = self.clob_api.get_orders(self.market.condition_id)
        print(f"order_dicts: {order_dicts}")
        self.orders = [
            Order(
                size=float(order_dict["size"]),
                price=float(order_dict["price"]),
                side=Side(order_dict["side"]),
                token=self.market.token(order_dict["token_id"]),
                size_matched=float(order_dict["size_matched"]),
                id=order_dict["id"],
            )
            for order_dict in order_dicts
        ]
        self.last_orders_time = time.time()
        return self.orders

    def get_order(self, order_id: str):
        order_dict = self.clob_api.get_order(order_id)
        if order_dict is None:
            return None

        size = float(order_dict.get("original_size", order_dict.get("size", 0)))
        token_id = int(order_dict.get("asset_id", order_dict.get("token_id", 0)))

        return Order(
                size=size,
                price=float(order_dict["price"]),
                side=Side(order_dict["side"]),
                token=self.market.token(token_id),
                size_matched=float(order_dict["size_matched"]),
                id=order_dict["id"],
            )
