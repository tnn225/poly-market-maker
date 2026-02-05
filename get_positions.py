import logging
import time

from poly_market_maker.utils import setup_logging
from poly_market_maker.clob_api import ClobApi
from poly_market_maker.utils.telegram import Telegram
from poly_market_maker.my_token import MyToken
from poly_market_maker.market import Market
from poly_market_maker.position_engine import PositionEngine
from poly_market_maker.constants import (
    WHITELIST,
    BLACKLIST,
    BRAIN_WALLETS,
    GOAT_WALLETS,
    in_whitelist,
    DEBUG,
    HOLDER_MIN_SIZE,
    MIN_TOTAL_SIZE,
    MAX_TRADED_COUNT,
)
# from poly_market_maker.price_engine import PriceEngine

setup_logging()
logger = logging.getLogger(__name__)

clob_api = ClobApi()
telegram = Telegram()
positions_engine = PositionEngine()

last_sizes = {}

SIDES = {MyToken.A: "Up", MyToken.B: "Down"}
SYMBOLS = ["btc"] # , "eth", "sol", "xrp"]

once = {}

def get(positions: list, key: str) -> float:
    return sum(float(position[key]) for position in positions)

def get_size(positions: list) -> float:
    return get(positions, "size")

def get_last_size(positions: list) -> float:
    return get(positions, "last_size")

def print_holders(interval: int, market: Market):
    global last_size
    holders_by_side = clob_api.get_holders(market)
    logger.info("Printing holders for interval: %s", interval)
    now = int(time.time())
    seconds_left = 900 - (now % 900)

    for my_token, side in SIDES.items():
        token_id = market.token_ids[my_token]
        holders = holders_by_side[side]
        for holder in holders:
            address = holder["proxyWallet"]
            if address in positions_engine.addresses and address not in positions_engine.active_addresses:
                positions_engine.active_addresses.append(address)

        positions = positions_engine.get_positions(token_id)
        size = get(positions,"size")
        last_size = last_sizes.get(side, 0)
        cost = get(positions,"cost")
        avg_price = cost / size if size > 0 else 0
        logger.info(f"side: {side} size: {size:.2f} ({size - last_size:+.2f}) shares")

        slug = f"btc-updown-15m-{interval}"

        if abs(size - last_size) < 0.00000001 and not DEBUG:
            logger.info(f"Skipping {slug} {side} {size:.2f} ({size - last_size:+.2f}) shares")
            continue

        num_wallets = len(positions)

        message = ""

        # price = bid if side == "Up" else (1 - ask)
        message = "\n".join([
            f"{num_wallets} sifu wallets with {size:.2f} ({size - last_size:+.2f}) shares {side} at ${avg_price:.2f} = ${cost:.2f}",
            f"<a href=\"https://polymarket.com/event/{slug}\">{slug}</a>",
            "",
            ""
        ])
        for position in positions:
            address = position["address"]
            name = position["name"]
            size = position["size"]
            last_size = position["last_size"]
            avg_price = position["avg_price"]
            cost = position["cost"]
            message += f"<a href=\"https://polymarket.com/profile/{address}\">{name}</a> {size:.2f} ({size - last_size:+.2f}) shares {side} at ${avg_price:.2f} = ${cost:.2f}\n"

        telegram.send_message(message, disable_web_page_preview=True)
        last_sizes[side] = size

def run(interval: int):
    global last_sizes
    end_time = interval + 900
    market = clob_api.get_market(interval)
    positions_engine.init_market(market)
    last_sizes = {}
    while time.time() < end_time:
        time.sleep(1)
        now = int(time.time()) + 10
        if now % 30 == 0:
            for symbol in SYMBOLS:
                try:
                    print_holders(interval, market)
                except Exception as e:
                    logger.error(f"Error printing holders: {e}")

def main():
    while True:
        time.sleep(1)
        interval = int(time.time()) // 900 * 900
        run(interval)

if __name__ == "__main__":
    main()
