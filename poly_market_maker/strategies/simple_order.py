import logging

from poly_market_maker.my_token import MyToken
from poly_market_maker.order import Order, Side


SIZE = 5
MAX_BALANCE = 10
MAX_IMBALANCE = 20
MAX_HEDGE_IMBALANCE = 50

class SimpleOrder:
    def __init__(self, token: MyToken):
        self.logger = logging.getLogger(self.__class__.__name__)

        assert isinstance(token, MyToken)

        # if config.spread >= config.depth:
        #    raise Exception("Depth does not exceed spread.")
        self.token = token
        self.p_min = 0.01
        self.p_max = 0.99
        self.delta = 0.01
        self.spread = 0.0
        self.depth = 10
        self.max_collateral = 100
        self.balance = 0
        self.imbalance = 0

    def set_balance(self, balance: float):
        self.balance = balance

    def set_imbalance(self, imbalance: float):
        self.imbalance = imbalance  

    def set_price(self, bid: float, delta: float):
        imbalance = self.imbalance - delta 
        self.buy_prices = []
        # print(f"  set_buy_prices self.depth: {self.depth} bid: {bid:.2f} imbalance: {imbalance:.2f}")
        for i in range(int(self.depth)):
            # price = min(round(bid - i * self.delta, 2), 0.49)
            price = round(bid - i * self.delta, 2) if bid >= 0.5 else round(bid - i * self.delta - self.spread, 2)
            if self.p_min <= price <= self.p_max:
                if imbalance > SIZE:
                    imbalance -= SIZE
                    continue
                self.buy_prices.append(price)

    def get_sell_orders(self):
        balance = self.balance
        orders = []
        for price in self.sell_prices:
            if SIZE <= balance:
                orders.append(
                    Order(
                        price=price,
                        side=Side.SELL,
                        token=self.token,
                        size=SIZE,
                    )
                )
                balance -= SIZE 
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

    def get_buy_order(self, price: float, size = SIZE):
        return Order(
            price=price,
            side=Side.BUY,
            token=self.token,
            size = size
        )

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


    def get_orders(self, seconds_left: int, delta: float, bid: float, ask: float, up: float):
        print(f"get_orders seconds_left: {seconds_left} delta: {delta:+.2f} bid: {bid:.2f} ask: {ask:.2f} up: {up:.2f}")
        self.up = up
        self.set_price(bid, delta)
        return self.get_buy_orders() # + self.get_sell_orders()
