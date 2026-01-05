from datetime import datetime, timezone
import logging
import os  
import time

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
from poly_market_maker.strategies.simple_strategy import SimpleStrategy

from dotenv import load_dotenv          # Environment variable management
load_dotenv()                           # Load environment variables from .env file

MAX_SHARES = 500
FUNDER = os.getenv("FUNDER")
TARGET = os.getenv("TARGET")

DEBUG = False
MIN_SIZE = 5

setup_logging()
logger = logging.getLogger(__name__)

clob_api = ClobApi()

price_engine = PriceEngine(symbol="btc/usd")
price_engine.start()

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

class TradeManager:
    def __init__(self, interval: int):
        self.clob_api = clob_api
        self.market = clob_api.get_market(interval)

        self.strategy = SimpleStrategy()
        self.order_book_engine = OrderBookEngine(self.market)
        self.order_book_engine.start()

        self.amm_a = SimpleOrder(token=MyToken.A)
        self.amm_b = SimpleOrder(token=MyToken.B)

        self.last_orders_time = 0
        self.orders = self.get_orders()

    def trade(self, seconds_left: int, delta: float):
        bid, ask = self.order_book_engine.get_bid_ask(MyToken.A)

        if bid is None or ask is None or (bid == 0 and ask == 0):
            print(f"{seconds_left} delta {delta:+.2f} Bid: {bid} Ask: {ask} - no bid or ask or both are 0")
            return

        up = bid
        down = round(1 - ask, 2)

        orders = [] 

        orders_a = self.amm_a.get_orders(seconds_left, 0, 0, bid, ask, up)
        orders_b = self.amm_b.get_orders(seconds_left, 0, 0, round(1 - ask, 2), round(1 - bid, 2), down)
   
        shares = self.balances[MyToken.A] + self.balances[MyToken.B]
        if shares < MAX_SHARES: 
            orders = orders_a + orders_b

        print(f"  Orders_a: {orders_a} Orders_b: {orders_b}")

        # Force refresh orders to get latest state before calculating what to cancel/place
        # self.last_orders_time = 0
        self.orders = self.get_orders()
        (orders_to_cancel, orders_to_place) = self.get_orders_to_cancel_and_place(orders)
        print(f"  Orders_to_cancel: {orders_to_cancel} Orders_to_place: {orders_to_place} Orders: {self.orders}")

        if not DEBUG and len(orders_to_cancel) + len(orders_to_place) > 0:
            self.cancel_orders(orders_to_cancel)
            self.place_orders(orders_to_place)
            self.last_orders_time = 0
            self.orders = self.get_orders()

    def cancel_orders(self, orders: list[Order]) -> list[Order]:
        order_ids = [order.id for order in orders]
        self.clob_api.cancel_orders(order_ids)
        return orders

    def place_orders(self, orders: list[Order]) -> list[Order]:
        # `py_clob_client` in this repo doesn't support PostOrdersArgs/batch posting.
        # Our `ClobApi.post_orders()` accepts dicts and posts one-by-one.
        orders_to_place = [
            {
                "price": order.price,
                "size": order.size,
                "side": order.side.value,
                "token_id": self.market.token_id(order.token),
            }
            for order in orders
        ]
        self.clob_api.post_orders(orders_to_place)

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

    @staticmethod
    def _new_order_from_order_type(order_type: OrderType, size: float) -> Order:
        return Order(
            price=order_type.price,
            size=size,
            side=order_type.side,
            token=order_type.token,
        )

    def get_balances(self) -> dict:
        balances = self.clob_api.get_balances(self.market)
        return balances 

    def get_orders(self) -> list[Order]:
        if time.time() - self.last_orders_time < 10:
            return self.orders
        
        self.balances = self.get_balances()
        self.amm_a.set_balance(self.balances[MyToken.A])
        self.amm_b.set_balance(self.balances[MyToken.B])
        self.amm_a.set_imbalance(self.balances[MyToken.A] - self.balances[MyToken.B])
        self.amm_b.set_imbalance(self.balances[MyToken.B] - self.balances[MyToken.A])
        self.last_orders_time = time.time()
        
        orders = self.clob_api.get_orders(self.market.condition_id)
        return [
            Order(
                size=order_dict["size"],
                price=order_dict["price"],
                side=Side(order_dict["side"]),
                token=self.market.token(order_dict["token_id"]),
                id=order_dict["id"],
            )
            for order_dict in orders
        ]

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

    def cancel_all_buy_orders(self, delta: float):
        for order in self.orders:
            if order.side == Side.BUY:
                self.clob_api.cancel_order(order.id)
  
def main():
    interval = 0 
    last = 0
    trade_manager = None

    while True:
        time.sleep(1)

        data = price_engine.get_data()
        if data is None:
            print("No price data")
            continue

        price = data.get('price')
        target = data.get('target')
        if price is None or target is None:
            continue
        delta = price - target


        now = int(time.time()) + 910
        seconds_left = 900 - (now % 900)
        if now // 900 * 900 > interval:  # 15-min intervals
            interval = now // 900 * 900
            if trade_manager is not None:
                trade_manager.cancel_all_buy_orders(delta)
                trade_manager.order_book_engine.stop() 

            interval = now // 900 * 900
            trade_manager = TradeManager(interval)

        trade_manager.trade(seconds_left, delta)

if __name__ == "__main__":
    main()
