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
        self.spread = 0.00
        self.depth = 10
        self.max_collateral = 100
        self.balance = 0
        self.imbalance = 0

    def set_balance(self, balance: float):
        self.balance = balance

    def set_imbalance(self, imbalance: float):
        self.imbalance = imbalance  

    def set_buy_prices(self, bid: float):
        self.buy_prices = []
        for i in range(int(self.depth)):
            # price = min(round(bid - i * self.delta, 2), 0.49)
            price = round(bid - i * self.delta, 2)
            if self.p_min <= price <= self.p_max: # and imbalance + SIZE < MAX_IMBALANCE:
                self.buy_prices.append(price)

    def set_sell_prices(self, ask: float):
        self.sell_prices = []
        for i in range(int(self.depth)):
            # price = max(round(self.up+self.spread, 2), round(ask + i * self.delta, 2))
            price = round(ask + i * self.delta, 2)
            if 0.01 <= price <= 0.99:
                self.sell_prices.append(price)

    def set_hedge_prices(self):
        self.hedge_prices = []
        for i in range(int(self.depth)):
            price = round(0.01 + i * self.depth, 2)
            if 0.01 <= price <= 0.1:
                self.hedge_prices.append(price)

    def set_price(self, bid: float, ask: float, up: float):
        self.up = up
        self.bid = bid 
        self.ask = ask 
        self.set_sell_prices(ask)
        self.set_hedge_prices()
        self.set_buy_prices(bid)
        print(f"  set_price {self.token} buy_prices={self.buy_prices} sell_prices={self.sell_prices} hedge_prices={self.hedge_prices}")

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

    def get_buy_order(self, price: float):
        return Order(
            price=price,
            side=Side.BUY,
            token=self.token,
            size = SIZE
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

    def get_skew(self, imbalance: float, price: float) -> float:
        # if price > 0.5:
        imbalance -= (price - 0.5) * 100 * 5
        if imbalance <= 0:
            return 0
        return (imbalance / SIZE) / 100

    def get_orders(self, seconds_left: int, price: float, delta: float, bid: float, ask: float, up: float):
        skew = self.get_skew(self.imbalance, bid)
        self.set_price(bid - skew, ask, up)

        return self.get_buy_orders() # + self.get_sell_orders()
