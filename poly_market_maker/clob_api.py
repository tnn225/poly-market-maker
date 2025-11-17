import logging
import os
import sys
import time
import requests

from poly_market_maker.market import Market
from py_clob_client.client import ClobClient, ApiCreds, OrderArgs, OpenOrderParams
from py_clob_client.exceptions import PolyApiException

from poly_market_maker.utils import randomize_default_price
from poly_market_maker.constants import OK
from poly_market_maker.metrics import clob_requests_latency

DEFAULT_PRICE = 0.5

from py_clob_client.client import ClobClient

from dotenv import load_dotenv
load_dotenv()

HOST = "https://clob.polymarket.com"
CHAIN_ID = 137
PRIVATE_KEY = os.getenv("PRIVATE_KEY")
FUNDER = os.getenv("FUNDER")

class ClobApi:
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.client = ClobClient(
            HOST,  # The CLOB API endpoint
            key=PRIVATE_KEY,  # Your wallet's private key
            chain_id=CHAIN_ID,  # Polygon chain ID (137)
            signature_type=1,  # 1 for email/Magic wallet signatures
            funder=FUNDER  # Address that holds your funds
        )
        self.client.set_api_creds(self.client.create_or_derive_api_creds())


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

    def get_orders(self, condition_id: str):
        """
        Get open keeper orders on the orderbook
        """
        self.logger.debug("Fetching open keeper orders from the API...")
        start_time = time.time()
        try:
            resp = self.client.get_orders(OpenOrderParams(market=condition_id))
            clob_requests_latency.labels(method="get_orders", status="ok").observe(
                (time.time() - start_time)
            )

            return [self._get_order(order) for order in resp]
        except Exception as e:
            self.logger.error(
                f"Error fetching keeper open orders from the CLOB API: {e}"
            )
            clob_requests_latency.labels(method="get_orders", status="error").observe(
                (time.time() - start_time)
            )
        return []

    def place_order(self, price: float, size: float, side: str, token_id: int) -> str:
        """
        Places a new order
        """
        self.logger.info(
            f"Placing a new order: Order[price={price},size={size},side={side},token_id={token_id}]"
        )
        start_time = time.time()
        try:
            resp = self.client.create_and_post_order(
                OrderArgs(price=price, size=size, side=side, token_id=token_id)
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
    

    def get_condition_id_by_slug(self, slug: str):
        url = f"https://gamma-api.polymarket.com/markets/slug/{slug}"
        resp = requests.get(url)

        if resp.status_code != 200:
            raise Exception(f"Error {resp.status_code}: {resp.text}")
                
        return resp.json().get("conditionId")


    def get_market(self, timestamp: int):
        slug = f"btc-updown-15m-{timestamp}"
        print( f"Fetching market for slug: {slug}")
        condition_id = self.get_condition_id_by_slug(slug)

        return Market(
            condition_id,
            self.client.get_collateral_address(),
        )