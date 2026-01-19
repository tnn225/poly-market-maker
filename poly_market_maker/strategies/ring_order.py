import logging

from poly_market_maker.my_token import MyToken
from poly_market_maker.order import Order, Side
from poly_market_maker.clob_api import ClobApi
from poly_market_maker.constants import MIN_SIZE, DEBUG
from poly_market_maker.market import Market

EPS = 0.0001

class RingOrder:
    def __init__(self, token: MyToken, market: Market):
        self.logger = logging.getLogger(self.__class__.__name__)

        assert isinstance(token, MyToken)
        self.clob_api = ClobApi()

        # if config.spread >= config.depth:
        #    raise Exception("Depth does not exceed spread.")
        self.token = token
        self.shares = {}
        self.sell_order_id = {}
        self.spread = 0.03

        self.balance = 0
        self.imbalance = 0
        self.other = None
        self.buy_order_id = None
        self.buy_order = None
        self.market = market

    def set_other(self, other):
        self.other = other

    def check_ring(self):
        self.logger.info(f"check_ring buy_order_id: {self.buy_order_id}")
        if self.buy_order_id is not None:
            self.buy_order = self.clob_api.get_order(self.buy_order_id)
            if self.buy_order is None or abs(self.buy_order['size_matched'] - MIN_SIZE) <= EPS:
                self.logger.info(f"Order {self.buy_order_id} is filled, removing from buy_order_id")
                self.buy_order_id = None
                self.buy_order = None
                self.other.shares[self.sell_price] = MIN_SIZE

    def is_removable(self, bid: float) -> bool:
        self.logger.info(f"is_removable bid: {bid} buy_order: {self.buy_order}")
        if self.buy_order is not None and float(self.buy_order['price']) + EPS < bid:
            return True
        return False

    def remove_ring(self, bid: float):
        if self.buy_order_id is not None:
            self.clob_api.cancel_order(self.buy_order_id)
            self.buy_order_id = None
            self.buy_order = None

    def has_ring(self, inventory: float, delta: float, bid: float, ask: float) -> bool:
        if inventory < 0 or (inventory == 0 and delta > 0):
            sell_price = 1 - ask - self.spread
            return 0.01 <= sell_price <= 0.99 and self.other.get_shares(sell_price) <= MIN_SIZE
        return False

    def add_ring(self, bid: float, ask: float):
        if self.buy_order is not None and abs(float(self.buy_order['price']) - bid) <= EPS:
            return

        # self.shares[bid] = MIN_SIZE
        self.buy_order_id = self.clob_api.place_order(bid, MIN_SIZE, Side.BUY.value, self.market.token_id(self.token))
        self.sell_price = round(1 - (bid + self.spread), 2)

    def get_shares(self, price: float) -> float:
        if price in self.shares:
            return self.shares[price]
        return 0.0

    def place_sell_order(self, bid: float):
        for price, order_id in list(self.sell_order_id.items()):
            sell_order = self.clob_api.get_order(order_id)
            if sell_order is None:
                del self.sell_order_id[price]
                self.logger.info(f"Order {order_id} not found, removing from sell_order_id")
                continue

            size = float(sell_order['size_matched'])
            if abs(size - MIN_SIZE) <= EPS: 
                del self.sell_order_id[price]
                self.logger.info(f"Order {order_id} is filled, removing from sell_order_id")
                continue

            if float(sell_order['price']) + EPS < bid:
                self.clob_api.cancel_order(order_id)
                del self.sell_order_id[price]
                self.shares[price] = MIN_SIZE
                self.logger.info(f"Order {order_id} is below bid, removing from sell_order_id")
                continue

        # Iterate through all shares and place sell orders
        for price, shares in list(self.shares.items()):
            if price >= bid and shares > EPS:
                order_id = self.clob_api.place_order(price, MIN_SIZE, Side.BUY.value, self.market.token_id(self.token))
                self.shares[price] = 0
                self.sell_order_id[price] = order_id
                self.logger.info(f"Placed sell order: {price} order_id: {order_id}")

    def trade(self, inventory: float, delta: float, bid: float, ask: float):
        self.logger.info(f"trade {self.token.value} inventory: {inventory:+.2f} delta: {delta:+.2f} bid: {bid:.2f}, ask: {ask:.2f}")

        self.check_ring()
        if not self.has_ring(inventory, delta, bid, ask) or self.is_removable(bid):
            self.remove_ring(bid)
        if self.has_ring(inventory, delta, bid, ask):
            self.add_ring(bid, ask)

        # Sell order
        # if inventory < 0:
        self.place_sell_order(bid)
