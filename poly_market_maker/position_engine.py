import csv
import os
import websocket
import threading
import json
import time
from datetime import datetime, timezone
import logging

from poly_market_maker.clob_api import ClobApi
from poly_market_maker.market import Market
from poly_market_maker.my_token import MyToken
from poly_market_maker.constants import DEBUG
from poly_market_maker.utils.common import get_sifu_addresses

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

ACTIVE_POLL_SECONDS = 30 if not DEBUG else 30
POLL_SECONDS = 300 if not DEBUG else 60

class PositionEngine():

    def __init__(self):
        self.clob_api = ClobApi()
        self.market = None
        self.addresses = get_sifu_addresses()
        self.active_addresses = []
        self.last_updated_times = {}

    def init_market(self, market: Market):
        self.market = market
        self.positions = {}
        self.last_positions = {}
        self.last_updated_times = {}
        self.active_addresses = []

    def is_polling(self, address: str) -> bool:
        if address not in self.last_updated_times:
            return True

        seconds = time.time() - self.last_updated_times[address]
        if address in self.active_addresses:
            return seconds > ACTIVE_POLL_SECONDS
        return seconds > POLL_SECONDS

    def poll(self, address: str):
        # if not self.is_polling(address):
        #    return

        # print(f"Polling {address}")
        self.last_positions[address] = self.positions.get(address, {})
        self.positions[address] = self.clob_api.get_positions(address, self.market)
        self.last_updated_times[address] = time.time()
        # print(f"Polled {address} {self.positions[address]}")

    def get_positions(self, token_id: int):
        positions = []
        for address in self.addresses:
            position = self.get_position(address, token_id)
            if position:
                positions.append(position)
        return sorted(positions, key=lambda p: p["size"], reverse=True)

    def get_position(self, address: str, token_id: int):
        self.poll(address)

        position = self.positions[address].get(token_id, None)
        if position is None:
            return None
        last_position = self.last_positions[address].get(token_id, None)

        position['last_size'] = last_position.get('size', 0) if last_position is not None else 0
        # if abs(position['size'] - position['last_size']) < 0.00000001:
        #     return None
        return position


# ==========================================================
#                 HOW TO USE THE ENGINE
# ==========================================================


def test_position_engine(interval: int):
    position_engine = PositionEngine(interval, "btc")

    market = position_engine.clob_api.get_market(interval)
    holders_by_side = position_engine.clob_api.get_holders(market)
    for side in holders_by_side:
        for holder in holders_by_side[side]:
            if len(position_engine.balances) > 2:
                continue
            position_engine.add_address(holder['proxyWallet'], position_engine.clob_api.get_balance(holder['proxyWallet']))

    
    end_time = interval + 900
    while time.time() < end_time:
        time.sleep(1)
        positions = position_engine.get_positions(market, market.token_ids[MyToken.A])
        for position in positions:
            logger.info(
                "%s %s (%.2f) at %.2f = $%.2f",
                position["address"],
                position["size"],
                position["size"] - position["last_size"],
                position["avg_price"],
                position["cost"],
            )

if __name__ == "__main__":
    now = int(time.time())
    interval = int(now // 900 * 900)

    test_position_engine(interval)
