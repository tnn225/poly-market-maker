import time
import logging


from poly_market_maker.utils import setup_logging
from poly_market_maker.clob_api import ClobApi
from poly_market_maker.utils.telegram import Telegram
from poly_market_maker.my_token import MyToken
from poly_market_maker.utils.common import get_sifu_addresses
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

emoji = '🧠'
setup_logging()
logger = logging.getLogger(__name__)

clob_api = ClobApi()
telegram = Telegram()
sifu_addresses = get_sifu_addresses()

 # engine = PriceEngine(symbol="btc/usd")

SIDES = {"Up": MyToken.A, "Down": MyToken.B}
SYMBOLS = ["btc"] # , "eth", "sol", "xrp"]

sifu_sizes = {}
last_sifu_sizes = {}
once = {}

def is_good(holder: dict) -> bool:
    if holder['proxyWallet'] in sifu_addresses:
        return True
    return holder['amount'] > HOLDER_MIN_SIZE

def only_once(slug: str, side: str, holders: list) -> bool:
    ret = False
    for holder in holders:
        if is_good(holder):
            print(f"{holder['name']} {holder['proxyWallet']} {holder['amount']:.2f} shares {side}")
        key = f"{slug}-{side}-{holder['proxyWallet']}"
        # print(f"key: {key}")
        if once.get(key):
            print(f"{key} in once")
        if is_good(holder) and not once.get(key):
            ret = True
            once[key] = True
    return ret

def get_size(holders: list) -> float:
    value = 0
    for holder in holders:
        value += float(holder['amount'])
    return value

def get_sifu_size(holders: list) -> float:
    size = 0
    for holder in holders:
        if holder['proxyWallet'] in sifu_addresses:
            size += float(holder["amount"])
    return size

def get_num_sifu_wallets(holders: list) -> int:
    num = 0
    for holder in holders:
        if holder['proxyWallet'] in sifu_addresses:
            num += 1
    return num

def format_diff(diff: float, use_html: bool = False) -> str:
    """Format difference with color/formatting if > 0"""
    formatted = f"{diff:+.0f}"
    if diff > 0:
        if use_html:
            return f"<b>{formatted}</b>"
        else:
            return f"\033[92m{formatted}\033[0m"  # ANSI green
    return formatted

def print_holders(interval: int, symbol: str = "btc"):
    global sifu_sizes, last_sifu_size
    now = int(time.time())
    seconds_left = 900 - (now % 900)

    market = clob_api.get_market(interval, symbol)

    holders_by_side = clob_api.get_holders(market)
    for side in holders_by_side:
        holders = holders_by_side[side]
        slug = f"{symbol}-updown-15m-{interval}"

        last_sifu_sizes[side] = sifu_sizes.get(side, 0)
        sifu_sizes[side] = get_sifu_size(holders)
        sifu_size = sifu_sizes[side]
        last_sifu_size = last_sifu_sizes[side]
        total_size = get_size(holders)
        num_sifu_wallets = get_num_sifu_wallets(holders)

        diff = sifu_size - last_sifu_size
        text = f"{emoji} {100*sifu_size / total_size:.0f}% = {sifu_size:.0f} ({format_diff(diff)}) / {total_size:.0f} shares {num_sifu_wallets} wallets {side}"

        if total_size < MIN_TOTAL_SIZE and (DEBUG or not only_once(slug, side, holders)) and num_sifu_wallets == 0:
            print(f"Skipping {slug} {text}")
            continue
        
        # price = bid if side == "Up" else (1 - ask)
        message = "\n".join([
            text,
            f"<a href=\"https://polymarket.com/event/{slug}\">{slug}</a>",
            "",
        ])

        for holder in holders:
            if not is_good(holder):
                continue

            sifu = '㊙️ ' if holder['proxyWallet'] in sifu_addresses else ''

            text = f"{sifu}<a href=\"https://polymarket.com/profile/{holder['proxyWallet']}\">{holder['name']}</a> {holder['amount']:.2f} shares {side}\n"
            message += text

        key = f"{slug}-{side}"
        if (total_size > MIN_TOTAL_SIZE and (DEBUG or not once.get(key, False))) or num_sifu_wallets > 0:
            once[key] = True
            telegram.send_message(message, disable_web_page_preview=True)
        print(message)


def run(interval: int):
    global sifu_sizes, last_sifu_size
    end_time = interval + 900
    market = clob_api.get_market(interval)
    last_sifu_size = {}
    sifu_sizes = {}
    while time.time() + 10 < end_time:
        try:
            print_holders(interval)
        except Exception as e:
            logger.error(f"Error printing holders: {e}")
        time.sleep(30)

def main():
    while True:
        time.sleep(1)
        interval = int(time.time()) // 900 * 900
        run(interval)

if __name__ == "__main__":
    main()
