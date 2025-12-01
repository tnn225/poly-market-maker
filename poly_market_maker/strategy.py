from enum import Enum
import json
import logging

from poly_market_maker.models import Model
from poly_market_maker.order_book_engine import OrderBookEngine
from poly_market_maker.orderbook import OrderBookManager
from poly_market_maker.price_engine import PriceEngine
from poly_market_maker.price_feed import PriceFeed
from poly_market_maker.prediction_engine import PredictionEngine
from poly_market_maker.my_token import MyToken, Collateral
from poly_market_maker.constants import MAX_DECIMALS

from poly_market_maker.strategies.base_strategy import BaseStrategy
from poly_market_maker.strategies.amm_strategy import AMMStrategy
from poly_market_maker.strategies.bands_strategy import BandsStrategy

from poly_market_maker.dataset import Dataset
from poly_market_maker.bucket_classifier import BucketClassifier

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

DEBUG = False

class Strategy(Enum):
    AMM = "amm"
    BANDS = "bands"

    @classmethod
    def _missing_(cls, value):
        if isinstance(value, str):
            for strategy in Strategy:
                if value.lower() == strategy.value.lower():
                    return strategy
        return super()._missing_(value)


class StrategyManager:
    def __init__(
        self,
        strategy: str,
        config_path: str,
        price_feed: PriceFeed,
        order_book_manager: OrderBookManager,
        price_engine: PriceEngine,
        order_book_engine: OrderBookEngine,
        prediction_engine: PredictionEngine,
    ) -> BaseStrategy:
        self.logger = logging.getLogger(self.__class__.__name__)

        with open(config_path) as fh:
            config = json.load(fh)

        self.price_feed = price_feed
        self.order_book_manager = order_book_manager
        self.price_engine = price_engine
        self.order_book_engine = order_book_engine
        self.prediction_engine = prediction_engine
        # dataset = Dataset()
        feature_cols = ['delta', 'percent', 'log_return', 'time', 'seconds_left', 'bid', 'ask']
        self.model = Model(f"RandomForestClassifier_1448_36_130_135_log2_False_balanced_subsample", RandomForestClassifier(n_estimators=253, max_depth=40, min_samples_split=54, min_samples_leaf=33, max_features=0.5, bootstrap=False, class_weight='balanced', random_state=42, n_jobs=-1), feature_cols=feature_cols)
        # self.model = Model(f"BucketClassifier", BucketClassifier(), feature_cols=feature_cols, dataset=dataset)

        match Strategy(strategy):
            case Strategy.AMM:
                self.strategy = AMMStrategy(config)
            case Strategy.BANDS:
                self.strategy = BandsStrategy(config)
            case _:
                raise Exception("Invalid strategy")

    def synchronize(self):
        self.logger.debug("Synchronizing strategy...")

        try:
            orderbook = self.get_order_book()
        except Exception as e:
            self.logger.error(f"{e}")
            return

        data = self.price_engine.get_data()
        price = data['price']
        target = data['target']
        timestamp = data['timestamp']
        seconds_left = 900 - timestamp % 900
        
        if price is None or target is None or seconds_left is None:
            self.logger.error(f"Price, target, or seconds_left is None")
            return
        
        bid, ask = self.order_book_engine.get_bid_ask(MyToken.A)
        if bid is None or ask is None:
            self.logger.error(f"Bid or ask is None")
            return
        
        up = self.model.get_probability(price, target, seconds_left, bid, ask)

        (orders_to_cancel, orders_to_place) = self.strategy.get_orders(orderbook, bid, ask, up, seconds_left)

        self.logger.debug(f"order existing: {len(orderbook.orders)}")
        self.logger.debug(f"order to cancel: {len(orders_to_cancel)}")
        self.logger.debug(f"order to place: {len(orders_to_place)}")

        if not DEBUG:
            self.cancel_orders(orders_to_cancel)
            self.place_orders(orders_to_place)

        self.logger.debug("Synchronized strategy!")

    def get_order_book(self):
        orderbook = self.order_book_manager.get_order_book()

        if None in orderbook.balances.values():
            self.logger.debug("Balances invalid/non-existent")
            raise Exception("Balances invalid/non-existent")

        if sum(orderbook.balances.values()) == 0:
            self.logger.debug("Wallet has no balances for this market")
            raise Exception("Zero Balances")

        return orderbook

    def get_token_prices(self):
        price_a = round(
            self.price_feed.get_price(MyToken.A),
            MAX_DECIMALS,
        )
        price_b = round(0.99 - price_a, MAX_DECIMALS)
        return {MyToken.A: price_a, MyToken.B: price_b}
    
    def get_bid_ask(self):
        return self.price_feed.get_bid_ask(MyToken.A)

    def cancel_orders(self, orders_to_cancel):
        if len(orders_to_cancel) > 0:
            self.logger.info(
                f"About to cancel {len(orders_to_cancel)} existing orders!"
            )
            self.order_book_manager.cancel_orders(orders_to_cancel)

    def place_orders(self, orders_to_place):
        if len(orders_to_place) > 0:
            self.logger.info(f"About to place {len(orders_to_place)} new orders!")
            self.order_book_manager.place_orders(orders_to_place)
