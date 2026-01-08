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

        self.amm_a = SimpleOrder(token=MyToken.A)
        self.amm_b = SimpleOrder(token=MyToken.B)

        self.last_orders_time = 0
        self.orders = self.get_orders()

    def cancel_orders(self, orders: list[Order]):
        for order in orders:
            self.clob_api.cancel_order(order.id)

    def place_orders(self, orders: list[Order]):
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

    def get_orders(self) -> list[Order]:
        if time.time() - self.last_orders_time < 10:
            return self.orders
        
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

    def trade(self, delta: float):
        if -1000 <= delta < -0:
            order = self.amm_a.get_buy_order(0.49)
            self.place_order(order)

        if 0 <= delta <= 1000:
            order = self.amm_b.get_buy_order(0.49)
            self.place_order(order)

  
def main():
    interval = 0 
    last = 0
    trade_manager = None
    has_orders = False

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

        now = int(time.time()) + 10
        seconds_left = 900 - (now % 900)
        print(f"seconds_left: {seconds_left} {price:.2f} {delta:+.2f}")
        if now // 900 * 900 > interval:  # 15-min intervals
            interval = now // 900 * 900
            trade_manager = TradeManager(interval)
            trade_manager.trade(delta)
            has_orders = True

        if trade_manager is not None and seconds_left < 600 and has_orders:
            trade_manager.last_orders_time = 0
            trade_manager.orders = trade_manager.get_orders()
            trade_manager.cancel_orders(trade_manager.orders)
            has_orders = False


if __name__ == "__main__":
    main()
