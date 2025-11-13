import logging
from math import sqrt

from poly_market_maker.token import Token, Collateral
from poly_market_maker.order import Order, Side
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
        depth: float,
        max_collateral: float,
    ):
        assert isinstance(p_min, float)
        assert isinstance(p_max, float)
        assert isinstance(delta, float)
        assert isinstance(depth, float)
        assert isinstance(spread, float)
        assert isinstance(max_collateral, float)

        self.p_min = p_min
        self.p_max = p_max
        self.delta = delta
        self.spread = spread
        self.depth = depth
        self.max_collateral = max_collateral


class AMM:
    def __init__(self, token: Token, config: AMMConfig):
        self.logger = logging.getLogger(self.__class__.__name__)

        assert isinstance(token, Token)

        # if config.spread >= config.depth:
        #    raise Exception("Depth does not exceed spread.")

        self.token = token
        self.p_min = config.p_min
        self.p_max = config.p_max
        self.delta = config.delta
        self.spread = config.spread
        self.depth = config.depth
        self.max_collateral = config.max_collateral

    def set_price(self, p_i: float):
        self.p_i = p_i
        bid = round(self.p_i, 2)
        self.ask = round(self.p_i + self.spread, 2)
        self.p_u = round(min(self.ask + self.depth, self.p_max), 2)
        self.p_l = round(max(bid - self.depth, self.p_min), 2)

        self.buy_prices = []
        logging.info(f"set_price p_i={self.p_i}, p_l={self.p_l}, p_u={self.p_u}, bid={bid}, ask={self.ask}")
        while self.p_l <= bid <= self.p_u:
            self.buy_prices.append(bid)
            bid = round(bid - self.delta, 2)
        logging.info(f"set_price p_i={self.p_i}, p_l={self.p_l}, p_u={self.p_u}, bid={bid}, ask={self.ask} buy_prices={self.buy_prices}")

    def get_sell_orders(self, balance):
        orders = []
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
                size=10 #  math_round_down(CAPITAL / bid, 2),  # size = $5 / price
            )
            for bid in self.buy_prices
        ]
        return orders

class AMMManager:
    def __init__(self, config: AMMConfig):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.amm_a = AMM(token=Token.A, config=config)
        self.amm_b = AMM(token=Token.B, config=config)
        self.max_collateral = config.max_collateral
        self.p_max = config.p_max

    def get_expected_orders(
        self,
        target_prices,
        balances,
        open_orders,
    ):
        if not target_prices[Token.A] or not target_prices[Token.B]:
            return []

        self.amm_a.set_price(target_prices[Token.A])
        self.amm_b.set_price(target_prices[Token.B])

        sell_orders_a = self.amm_a.get_sell_orders(balances[Token.A])
        sell_orders_b = self.amm_b.get_sell_orders(balances[Token.B])

        buy_orders_a = self.amm_a.get_buy_orders()
        buy_orders_b = self.amm_b.get_buy_orders()

        # assert len(buy_orders_a) == len(buy_orders_b)
        
        all_orders = set(OrderType(order) for order in open_orders)

        orders = []

        if target_prices[Token.A] >= 0.70:
            return sell_orders_a + sell_orders_b + buy_orders_a
        
        if target_prices[Token.B] >= 0.70:
            return sell_orders_a + sell_orders_b + buy_orders_b
        
        for i in range(min(len(buy_orders_a), len(buy_orders_b))):
            buy_order_a = buy_orders_a[i]
            buy_order_b = buy_orders_b[i]

            order_type_a = OrderType(buy_order_a)
            order_type_b = OrderType(buy_order_b)

            if order_type_a not in all_orders and order_type_b not in all_orders:
                orders.append(buy_order_a)
                orders.append(buy_order_b)

        return sell_orders_a + sell_orders_b + orders