import logging
import os
import time
import requests

from poly_market_maker.my_token import MyToken
from poly_market_maker.market import Market
from poly_market_maker.order import Order
from poly_market_maker.utils.common import randomize_default_price
from py_clob_client.client import ClobClient, ApiCreds, OrderArgs, OpenOrderParams
from py_clob_client.clob_types import OrderType as CLOBOrderType
from py_clob_client.exceptions import PolyApiException

from poly_market_maker.constants import OK, DEBUG
from poly_market_maker.metrics import clob_requests_latency

DEFAULT_PRICE = 0.5

from dotenv import load_dotenv
load_dotenv()

from poly_market_maker.utils.telegram import Telegram

telegram = Telegram()

HOST = "https://clob.polymarket.com"
CHAIN_ID = 137
PRIVATE_KEY = os.getenv("PRIVATE_KEY")
FUNDER = os.getenv("FUNDER")

# Maker fee rate in basis points. Must match the market's configured maker fee.
# Defaulting to 1000 bps based on observed API error; override via env if needed.
FEE_RATE_BPS = int(os.getenv("FEE_RATE_BPS", "1000"))

class ClobApi:
    """
    Singleton-ish API wrapper.

    `ClobApi()` will always return the same instance and will only initialize the
    underlying `ClobClient` once. This prevents repeated API key derivation when
    multiple modules instantiate `ClobApi`.
    """

    _instance = None
    _initialized = False

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if self.__class__._initialized:
            return
        self.logger = logging.getLogger(self.__class__.__name__)
        self.fee_rate_bps = FEE_RATE_BPS

        self.last_orders_time = 0
        self.orders = []
        self.markets = {}

        
        # Turn off DEBUG logging for ClobClient and underlying HTTP clients
        # Configure before ClobClient instantiation to catch all initialization logs
        http_loggers = [
            "py_clob_client",
            "httpx",
            "httpcore",
            "httpcore.http2",
            "httpcore.http11",
            "h2",
            "h2.connection",
            "h2.streams",
            "h2.events",
            "h2.settings",
            "h2.frame_buffer",
            "hpack",
            "hpack.huffman",
            "hpack.table",
            "urllib3",
            "urllib3.connectionpool",
        ]
        for logger_name in http_loggers:
            logging.getLogger(logger_name).setLevel(logging.ERROR)
        
        self.client = ClobClient(
            HOST,  # The CLOB API endpoint
            key=PRIVATE_KEY,  # Your wallet's private key
            chain_id=CHAIN_ID,  # Polygon chain ID (137)
            signature_type=1,  # 1 for email/Magic wallet signatures
            funder=FUNDER  # Address that holds your funds
        )

        self.client.set_api_creds(self.client.create_or_derive_api_creds())
        self.__class__._initialized = True

    def get_address(self):
        return self.client.get_address()

    def get_collateral_address(self):
        return self.client.get_collateral_address()

    def get_conditional_address(self):
        return self.client.get_conditional_address()

    def get_exchange(self, neg_risk = False):
        return self.client.get_exchange_address(neg_risk)

    def get_price(self, token_id: int) -> float:
        """
        Get the current price on the orderbook
        """
        self.logger.debug("Fetching midpoint price from the API...")
        start_time = time.time()
        try:
            resp = self.client.get_midpoint(token_id)
            clob_requests_latency.labels(method="get_midpoint", status="ok").observe(
                (time.time() - start_time)
            )
            if resp.get("mid") is not None:
                return float(resp.get("mid"))
        except Exception as e:
            self.logger.error(f"Error fetching current price from the CLOB API: {e}")
            clob_requests_latency.labels(method="get_midpoint", status="error").observe(
                (time.time() - start_time)
            )

        return self._rand_price()

    def _rand_price(self) -> float:
        price = randomize_default_price(DEFAULT_PRICE)
        self.logger.info(
            f"Could not fetch price from CLOB API, returning random price: {price}"
        )
        return price

    def get_order(self, order_id: str):
        try:
            order = self.client.get_order(order_id)
            order['size_matched'] = float(order['size_matched'])
            order['price'] = round(float(order['price']), 2)
            return order
        except Exception as e:
            self.logger.error(f"Error fetching order {order_id}: {e}")
            return None

    def get_orders(self, condition_id: str):
        """
        Get open keeper orders on the orderbook
        """
        self.logger.debug("Fetching open keeper orders from the API...")
        start_time = time.time()
        if time.time() - self.last_orders_time < 10:
            return self.orders
        self.last_orders_time = time.time()
        try:
            resp = self.client.get_orders(OpenOrderParams(market=condition_id))
            clob_requests_latency.labels(method="get_orders", status="ok").observe(
                (time.time() - start_time)
            )
            self.orders = [self._get_order(order) for order in resp]
        except Exception as e:
            self.logger.error(f"Error fetching keeper open orders from the CLOB API: {e}")
            clob_requests_latency.labels(method="get_orders", status="error").observe((time.time() - start_time))
            self.orders = []
        return self.orders

    def place_order(self, price: float, size: float, side: str, token_id: int) -> str:
        """
        Places a new order
        """
        self.logger.info(
            f"Placing a new order: Order[price={price},size={size},side={side},token_id={token_id}]"
        )
        if DEBUG:
            return None

        start_time = time.time()
        try:
            resp = self.client.create_and_post_order(
                OrderArgs(
                    price=price,
                    size=size,
                    side=side,
                    token_id=str(token_id),
                    fee_rate_bps=self.fee_rate_bps,
                )
            )
            clob_requests_latency.labels(
                method="create_and_post_order", status="ok"
            ).observe((time.time() - start_time))
            order_id = None
            if resp and resp.get("success") and resp.get("orderID"):
                order_id = resp.get("orderID")
                self.logger.info(
                    f"Succesfully placed new order: Order[id={order_id},price={price},size={size},side={side},tokenID={token_id}]!"
                )
                return order_id

            err_msg = resp.get("errorMsg")
            self.logger.error(
                f"Could not place new order! CLOB returned error: {err_msg}"
            )
        except Exception as e:
            self.logger.error(f"Request exception: failed placing new order: {e}")
            clob_requests_latency.labels(
                method="create_and_post_order", status="error"
            ).observe((time.time() - start_time))
        return None

    def place_orders(self, orders: list) -> list:
        """
        Place multiple orders using `post_orders`, which signs and posts each one.

        Accepts either:
        - poly_market_maker.order.Order objects (will be converted), or
        - dicts with keys: price, size, side, token_id, optional fee_rate_bps.
        """
        if not orders:
            return []

        # Place the first order immediately
        self.place_order(orders[0])
        orders = orders[1:]

        if not orders:
            return []

        if len(orders) > 15:
            self.place_orders(orders[:15])
            self.place_orders(orders[15:])
            return []

        payloads = []
        for idx, order in enumerate(orders):
            if isinstance(order, Order):
                payloads.append(
                    {
                        "price": order.price,
                        "size": order.size,
                        "side": order.side.value,
                        "token_id": order.token if isinstance(order.token, str) else order.token.value if hasattr(order.token, "value") else order.token,
                    }
                )
            elif isinstance(order, dict):
                payloads.append(order)
            else:
                self.logger.error(
                    f"Unsupported order type at position {idx}: {order}"
                )

        return self.post_orders(payloads)

    def cancel_order(self, order_id) -> bool:
        self.logger.info(f"Cancelling order {order_id}...")
        if order_id is None:
            self.logger.debug("Invalid order_id")
            return True

        start_time = time.time()
        try:
            resp = self.client.cancel(order_id)
            clob_requests_latency.labels(method="cancel", status="ok").observe(
                (time.time() - start_time)
            )
            return resp == OK
        except Exception as e:
            self.logger.error(f"Error cancelling order: {order_id}: {e}")
            clob_requests_latency.labels(method="cancel", status="error").observe(
                (time.time() - start_time)
            )
        return False

    def cancel_all_orders(self) -> bool:
        self.logger.info("Cancelling all open keeper orders..")
        start_time = time.time()
        try:
            resp = self.client.cancel_all()
            clob_requests_latency.labels(method="cancel_all", status="ok").observe(
                (time.time() - start_time)
            )
            return resp == OK
        except Exception as e:
            self.logger.error(f"Error cancelling all orders: {e}")
            clob_requests_latency.labels(method="cancel_all", status="error").observe(
                (time.time() - start_time)
            )
        return False

    def cancel_orders(self, order_ids: list[str]) -> bool:
        """
        Cancel multiple orders.
        (Convenience wrapper around py_clob_client's `client.cancel_orders()`.)
        """
        self.logger.info(f"Cancelling {len(order_ids)} orders...")
        if not order_ids:
            return True
        try:
            resp = self.client.cancel_orders(order_ids)
            return resp == OK
        except Exception as e:
            self.logger.error(f"Error cancelling orders: {e}")
            return False

    def _get_order(self, order_dict: dict) -> dict:
        size = float(order_dict.get("original_size")) - float(
            order_dict.get("size_matched")
        )
        price = float(order_dict.get("price"))
        side = order_dict.get("side")
        order_id = order_dict.get("id")
        token_id = int(order_dict.get("asset_id"))

        return {
            "size": size,
            "price": price,
            "side": side,
            "token_id": token_id,
            "id": order_id,
        }
    
    def get_bids_asks(self, token_id: int):
        orderbook = self.client.get_order_book(token_id) 
        # self.logger.debug(f"orderbook {orderbook}")

        # Convert to list of tuples (price, size) as floats
        bids = [(float(b.price), float(b.size)) for b in orderbook.bids]
        asks = [(float(a.price), float(a.size)) for a in orderbook.asks]

        # Sort bids descending (highest bid first), asks ascending (lowest ask first)
        bids.sort(key=lambda x: x[0], reverse=True)
        asks.sort(key=lambda x: x[0])
        return bids, asks
    
    def get_bid_ask(self, token_id: int):
        """Return best bid/ask from cached values."""
        bids, asks = self.get_bids_asks(token_id)
        best_bid = bids[0][0] if bids else None
        best_ask = asks[0][0] if asks else None
        return best_bid, best_ask

    def get_condition_id_by_slug(self, slug: str):
        url = f"https://gamma-api.polymarket.com/markets/slug/{slug}"
        resp = requests.get(url)

        if resp.status_code != 200:
            raise Exception(f"Error {resp.status_code}: {resp.text}")
                
        return resp.json().get("conditionId")

    def get_market(self, timestamp: int, symbol: str='btc'):
        slug = f"{symbol}-updown-15m-{timestamp}"
        if slug in self.markets:
            return self.markets[slug]
        print( f"Fetching market for slug: {slug}")
        condition_id = self.get_condition_id_by_slug(slug)
        self.markets[slug] = Market(condition_id, self.client.get_collateral_address())
        return self.markets[slug]

    def get_balances(self, market: Market, user: str = FUNDER):
        params = {
            "sizeThreshold": 1,
            "limit": 100,
            "sortBy": "TOKENS",
            "sortDirection": "DESC",
            "user": user,
            "market": market.condition_id,
        }
        url = "https://data-api.polymarket.com/positions"
        response = requests.get(url, params=params)

        response.raise_for_status()

        # print(f"response.json(): {response.json()}")
        positions = self._parse_positions(response.json())
        # print(f"positions: {positions}")
        # print(market.token_ids[MyToken.A], market.token_ids[MyToken.B])
        balances = {
            MyToken.A: positions.get(market.token_ids[MyToken.A]).get('size') if positions.get(market.token_ids[MyToken.A]) else 0,
            MyToken.B: positions.get(market.token_ids[MyToken.B]).get('size') if positions.get(market.token_ids[MyToken.B]) else 0,
        }
        self.logger.info(f"balances: {balances}")
        return balances

    def _parse_positions(self, positions: list) -> list:
        """Parse and format position data."""

        parsed = {}
        for pos in positions:
            asset_id = int(pos.get('asset'))
            parsed[asset_id] = {
                'outcome': pos.get('outcome'),
                'size': float(pos.get('size')),
                'avg_price': pos.get('avgPrice'),
                'current_price': pos.get('curPrice'),
                'initial_value': pos.get('initialValue'),
                'current_value': pos.get('currentValue'),
                'cash_pnl': pos.get('cashPnl'),
                'percent_pnl': pos.get('percentPnl'),
                'slug': pos.get('slug'),
                'title': pos.get('title'),
                'end_date': pos.get('endDate'),
            }
        return parsed

    def get_holders(self, market: Market):
        url = f"https://data-api.polymarket.com/holders?market={market.condition_id}"
        response = requests.get(url)
        response.raise_for_status()
        return response.json()


def main():
    clob_api = ClobApi()
    market = clob_api.get_market(1768463100)
    user = "0xe01ae9d586b428c251043368f808e678d4c4132c" 
    positions = clob_api.get_balances(market, user)
    print(f"Positions: {positions}")

if __name__ == "__main__":
    main()
