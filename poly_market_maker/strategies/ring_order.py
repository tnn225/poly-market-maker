import logging

from poly_market_maker.my_token import MyToken
from poly_market_maker.order import Order, Side
from poly_market_maker.clob_api import ClobApi
from poly_market_maker.constants import MIN_SIZE, DEBUG


EPS = 0.0001

class RingOrder:
    def __init__(self, token: MyToken):
        self.logger = logging.getLogger(self.__class__.__name__)

        assert isinstance(token, MyToken)
        self.clob_api = ClobApi()

        # if config.spread >= config.depth:
        #    raise Exception("Depth does not exceed spread.")
        self.token = token
        self.shares = {}
        self.pair_order = None 

        self.balance = 0
        self.imbalance = 0
        self.other = None

    def set_other(self, other):
        self.other = other

    def check_ring(self):
        if self.buy_order is not None:
            order = self.clob_api.get_order(self.buy_order.id)
            size = float(order['size_matched'])
            if abs(size - MIN_SIZE) <= EPS:
                self.buy_order = None
                self.shares[self.sell_price] = size
 
    def has_ring(self, inventory: float, delta: float, bid: float, ask: float) -> bool:
        if inventory < 0 or (inventory == 0 and delta >= 0):
            return self.get_shares(1-ask - self.spread) <= EPS
        return False

    def add_ring(self, bid: float, ask: float):
        self.shares[bid] = MIN_SIZE
        order = Order(price=bid, side=Side.BUY, token=self.token, size=MIN_SIZE)
        self.other.buy_order = self.clob_api.place_order(order)
        self.other.sell_price = 1 - ask - self.spread

    def get_shares(self, price: float) -> bool:
        if price in self.shares:
            return self.shares[price]
        return 0

    def place_order(self, bid: float):
        shares = self.get_shares(bid)
        if shares > EPS:
            order = Order(price=bid, side=Side.BUY, token=self.token, size=shares)
            self.clob_api.place_order(order)
            self.shares[bid] = 0

