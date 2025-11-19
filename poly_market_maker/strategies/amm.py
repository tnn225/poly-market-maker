import logging
from math import sqrt

from poly_market_maker.my_token import MyToken, Collateral
from poly_market_maker.order import Order, Side
from poly_market_maker.orderbook import OrderBook
from poly_market_maker.utils import math_round_down

CAPITAL = 5

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

    def set_price(self, bid: float, ask: float, up: float):
        self.bid = bid
        self.ask = ask 
        self.buy_prices = []
        self.sell_prices = [] 
        for i in range(int(self.depth)):
            price = round(bid - i * self.delta, 2)
            if self.p_min <= price <= self.p_max and price <= up - self.spread:
                self.buy_prices.append(price)
        logging.info(f"set_price bid={bid}, ask={ask} buy_prices={self.buy_prices}")

    def get_sell_orders(self, balance):
        orders = []
        return orders
        price = self.ask + self.depth
        size = 10
        max_price = min(0.99, self.ask + self.depth)
        while price <= max_price and size <= balance:
            orders.append(
                Order(
                    price=price,
                    side=Side.SELL,
                    token=self.token,
                    size=size,
                )
            )
            price = round(price + self.delta, 2)
            balance -= size
        return orders

    def get_buy_orders(self):
        """Return buy orders with fixed capital per level"""
        orders = [
            Order(
                price=bid,
                side=Side.BUY,
                token=self.token,
                # size=math_round_down(CAPITAL / bid, 2),  
                size = 5
            )
            for bid in self.buy_prices
        ]
        return orders
    
    def hedge_order(self, size):
        if self.bid <= 0.01: 
            self.bid = 0.01
        # if self.bid < 0.05: 
        return Order(
            price=self.bid,
            side=Side.BUY,
            token=self.token,
            size = 1 / self.bid
        )

class AMMManager:
    def __init__(self, config: AMMConfig):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.amm_a = AMM(token=MyToken.A, config=config)
        self.amm_b = AMM(token=MyToken.B, config=config)
        self.max_collateral = config.max_collateral
        self.p_max = config.p_max
        self.spread = config.spread

    def get_expected_orders(self, orderbook: OrderBook, bid: float, ask: float, up: float):
        if not bid or not ask:
            return []
        
        open_orders = orderbook.orders
        balances = orderbook.balances

        down = 1.0 - up
        self.amm_a.set_price(bid, ask, up)
        self.amm_b.set_price(round(1 - ask, 2), round(1 - bid,2), down)

        # sell_orders_a = self.amm_a.get_sell_orders(balances[MyToken.A])
        # sell_orders_b = self.amm_b.get_sell_orders(balances[MyToken.B])

        buy_orders_a = self.amm_a.get_buy_orders()
        buy_orders_b = self.amm_b.get_buy_orders()

        # assert len(buy_orders_a) == len(buy_orders_b)
        
        all_orders = set(OrderType(order) for order in open_orders)

        orders = []

        if balances[MyToken.A] <= max(balances[MyToken.B] + 50, 150):
            for order in buy_orders_a:
                # if OrderType(order) not in all_orders:
                orders.append(order)
        
        if balances[MyToken.B] <= max(balances[MyToken.A] + 50, 150):
            for order in buy_orders_b:
                # if OrderType(order) not in all_orders:
                orders.append(order)

        HEDGE_PRICE = 0.1
        HEDGE_SIZE = 10
        if bid < HEDGE_PRICE and balances[MyToken.A] < balances[MyToken.B] :
            # print('heging... A')
            orders.append(self.amm_a.hedge_order(HEDGE_SIZE))

        if (1 - ask) < HEDGE_PRICE and balances[MyToken.B] < balances[MyToken.A]:
            # print('heging... B')
            orders.append(self.amm_b.hedge_order(HEDGE_SIZE))

        return orders
    
