import logging
import time

from poly_market_maker.order import Order
from poly_market_maker.price_engine import PriceEngine
from poly_market_maker.my_token import MyToken
from poly_market_maker.utils import setup_logging
from poly_market_maker.clob_api import ClobApi
from poly_market_maker.order import Side
from poly_market_maker.strategies.simple_order import SimpleOrder

MAX_SHARES = 500
MIN_SIZE = 200

setup_logging()

clob_api = ClobApi()

price_engine = PriceEngine(symbol="btc/usd")

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
            if order.size == MIN_SIZE:
                self.clob_api.cancel_order(order.id)

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
        if -1000 <= delta < -500:
            order = self.amm_a.get_buy_order(0.49, MIN_SIZE)
            print(f"trade delta: {delta} order: {order}")
            self.place_order(order)

        if 500 <= delta <= 1000:
            order = self.amm_b.get_buy_order(0.49, MIN_SIZE)
            print(f"trade delta: {delta} order: {order}")
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
