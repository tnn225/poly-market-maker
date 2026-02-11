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

emoji_sifu = '🧠'
emoji_fresh = '🐥'

setup_logging()
logger = logging.getLogger(__name__)

clob_api = ClobApi()
telegram = Telegram()
sifu_addresses = get_sifu_addresses()
print(f"sifu_addresses: {sifu_addresses}")

 # engine = PriceEngine(symbol="btc/usd")

SIDES = {"Up": MyToken.A, "Down": MyToken.B}
SYMBOLS = ["btc"] # , "eth", "sol", "xrp"]

sifu_sizes = {}
last_sifu_sizes = {}
fresh_sizes = {}
last_fresh_sizes = {}
once = {}

def is_good(holder: dict) -> bool:
    if holder['proxyWallet'] in sifu_addresses:
        return True
    return holder['amount'] > HOLDER_MIN_SIZE or holder['trades'] < 30

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

def get_fresh_size(holders: list) -> float:
    size = 0
    for holder in holders:
        if holder['trades'] < 30:
            size += float(holder["amount"])
    return size

def get_num_fresh_wallets(holders: list) -> int:
    num = 0
    for holder in holders:
        if holder['trades'] < 30:
            num += 1
    return num

def format_diff(diff: float) -> str:
    """Format difference with color/formatting if > 0"""
    formatted = f"{diff:+.0f}"
    if diff > 0:
        return f"<b>{formatted}🟢</b>"
    return formatted

def print_holders(interval: int, symbol: str = "btc"):
    global sifu_sizes, last_sifu_size
    now = int(time.time())
    seconds_left = 900 - (now % 900)

    if seconds_left > 300:
        print(f"Skipping {interval} {symbol} {seconds_left} seconds left")
        return

    market = clob_api.get_market(interval, symbol)

    holders_by_side = clob_api.get_holders(market)
    for side in holders_by_side:
        holders = holders_by_side[side]
        slug = f"{symbol}-updown-15m-{interval}"

        total_size = get_size(holders)
        num_sifu_wallets = get_num_sifu_wallets(holders)
        last_sifu_sizes[side] = sifu_sizes.get(side, 0)
        sifu_sizes[side] = get_sifu_size(holders)
        sifu_size = sifu_sizes[side]
        last_sifu_size = last_sifu_sizes[side]
        diff = sifu_size - last_sifu_size

        sifu_text = f"{emoji_sifu} {100*sifu_size / total_size:.0f}% = {sifu_size:.0f} ({format_diff(diff)}) / {total_size:.0f} shares {num_sifu_wallets} wallets"

        num_fresh_wallets = get_num_fresh_wallets(holders)
        fresh_size = get_fresh_size(holders)
        fresh_text = f"{emoji_fresh} {100*fresh_size / total_size:.0f}% = {fresh_size:.0f} / {total_size:.0f} shares {num_fresh_wallets} wallets"

        if total_size < MIN_TOTAL_SIZE and only_once(slug, side, holders) and num_sifu_wallets == 0:
            print(f"Skipping {slug} {side} \n {fresh_text} \n {sifu_text}")
            continue
        
        # price = bid if side == "Up" else (1 - ask)
        message = "\n".join([
            fresh_text,
            f"\n{sifu_text}\n" if num_sifu_wallets > 0 else "",
            f"<a href=\"https://polymarket.com/event/{slug}\">{slug}</a> {side}",
            "",
            "",
        ])

        for holder in holders:
            if not is_good(holder):
                continue

            emoji = ""
            emoji += f'{emoji_sifu} ' if holder['proxyWallet'] in sifu_addresses else ''
            emoji += f'{emoji_fresh} ' if holder['trades'] < 30 else ''
            text = f"{emoji}<a href=\"https://polymarket.com/profile/{holder['proxyWallet']}\">{holder['name']}</a> {holder['amount']:.2f} shares\n"
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
        time.sleep(10)

def main():
    while True:
        time.sleep(1)
        interval = int(time.time()) // 900 * 900
        run(interval)

if __name__ == "__main__":
    main()
