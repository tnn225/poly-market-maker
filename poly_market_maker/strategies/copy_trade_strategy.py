from poly_market_maker.order import Order
from poly_market_maker.my_token import MyToken
from poly_market_maker.order import Side
import time
from poly_market_maker.strategies.base_strategy import BaseStrategy


ORDER_SIZE = 100
DEBUG = False
ADDRESS = "0xcc553b67cfa321f74c56515727ebe16dcb137cb3"
# ADDRESS = "0xe041d09715148a9a4a7a881a5580da2c0701f2e5"

# ADDRESS = "0x506bce138df20695c03cd5a59a937499fb00b0fe"

class CopyTradeStrategy(BaseStrategy):
    def __init__(self, interval: int):
        super().__init__(interval)
        self.buy_token_a = True
        self.buy_token_b = True
        self.buy_prices = [0.05, 0.10, 0.15]

    def trade(self):
        balances = self.clob_api.get_balances(self.market, ADDRESS)
        if not((balances[MyToken.A] > 0 and self.buy_token_a) or (balances[MyToken.B] > 0 and self.buy_token_b)):
            self.logger.debug(f"no trade opportunities")
            return

        bid, ask = self.order_book_engine.get_bid_ask(MyToken.A)
        self.logger.info(f"bid: {bid}, ask: {ask}")

        if bid is None or ask is None:
            self.logger.debug(f"No bid or ask")
            return

        orders = []
        if balances[MyToken.A] > 0 and self.buy_token_a:
            self.buy_token_a = False
            for buy_price in self.buy_prices:
                orders.append(Order(price=min(bid, buy_price), size=ORDER_SIZE, side=Side.BUY, token=MyToken.A))
        if balances[MyToken.B] > 0 and self.buy_token_b:
            self.buy_token_b = False
            for buy_price in self.buy_prices:
                orders.append(Order(price=min(1-ask,buy_price), size=ORDER_SIZE, side=Side.BUY, token=MyToken.B))
        if not DEBUG and len(orders) > 0:
            self.place_orders(orders)
        self.logger.info(f"Placed orders: {orders}")

    def run(self):
        while int(time.time()) <= self.end_time:
            time.sleep(10)
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
        self.logger.info("Copy trade strategy finished running")

