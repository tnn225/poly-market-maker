import time
from enum import Enum
import logging

from poly_market_maker.clob_api import ClobApi
from poly_market_maker.market import Market
from poly_market_maker.my_token import MyToken


MAX_SECONDS = 1 


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

        # token_id -> timestamp
        self.timestamp = {}

        # token_id -> cached data
        self.bids = {}
        self.asks = {}

    # -------------------------------
    # Internal: load orderbook if stale
    # -------------------------------
    def _ensure_cache(self, token: MyToken):
        token_id = self.market.token_id(token)

        now = time.time()
        last = self.timestamp.get(token_id, 0)

        if now - last > MAX_SECONDS:
            # Cache expired → fetch new orderbook
            bids, asks = self.clob_api.get_bids_asks(token_id)

            self.bids[token_id] = bids
            self.asks[token_id] = asks
            self.timestamp[token_id] = now

            # self.logger.debug(f"Refreshed cache for token {token_id}")
        else:
            pass
            # self.logger.debug(f"Using cached bids/asks for token {token_id}")

    # -------------------------------
    # Public API
    # -------------------------------
    def get_price(self, token: MyToken) -> float:
        """Always fetch fresh midpoint price (not cached)."""
        token_id = self.market.token_id(token)
        return self.clob_api.get_price(token_id)

    def get_bids_asks(self, token: MyToken):
        """Return cached bids/asks, refresh if stale."""
        self._ensure_cache(token)
        token_id = self.market.token_id(token)
        return self.bids[token_id], self.asks[token_id]



