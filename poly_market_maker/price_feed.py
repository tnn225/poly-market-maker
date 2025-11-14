from enum import Enum
import logging

from poly_market_maker.clob_api import ClobApi
from poly_market_maker.market import Market
from poly_market_maker.my_token import MyToken


class PriceFeedSource(Enum):
    CLOB = "clob"


class PriceFeed:
    """Market mid price resolvers"""

    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)

    def get_price(self) -> float:
        raise NotImplemented()


class PriceFeedClob(PriceFeed):
    """Resolves the prices from the clob"""

    def __init__(self, market: Market, clob_api: ClobApi):
        super().__init__()

        assert isinstance(market, Market)
        assert isinstance(clob_api, ClobApi)

        self.market = market
        self.clob_api = clob_api

    def get_price(self, token: MyToken) -> float:
        token_id = self.market.token_id(token)
        # self.logger.debug("Fetching target price using the clob midpoint price...")
        target_price = self.clob_api.get_price(token_id)
        # target_price, best_ask = self.get_bid_ask(token)
        # self.logger.debug(f"target_price: {target_price}")
        return target_price

    def get_bids_asks(self, token: MyToken):
        token_id = self.market.token_id(token)
        # self.logger.debug("Fetching bids and asks from the clob...")
        bids, asks = self.clob_api.get_bids_asks(token_id)
        # self.logger.debug(f"bids: {bids}, asks: {asks}")
        return bids, asks
    
    def get_bid_ask(self, token: MyToken):
        bids, asks = self.get_bids_asks(token)
        best_bid = bids[0][0] if len(bids) else None
        best_ask = asks[0][0] if len(asks) else None
        return best_bid, best_ask