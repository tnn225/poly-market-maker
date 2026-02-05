import time

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

setup_logging()

clob_api = ClobApi()
telegram = Telegram()
sifu_addresses = get_sifu_addresses()

 # engine = PriceEngine(symbol="btc/usd")

SIDES = {MyToken.A: "Up", MyToken.B: "Down"}
SYMBOLS = ["btc"] # , "eth", "sol", "xrp"]

sifu_size = 0
last_sifu_size = 0
once = {}

def is_good(holder: dict) -> bool:
    if holder['proxyWallet'] in sifu_addresses:
        return True
    if holder['amount'] < 1000:
        return False
    return holder['amount'] > HOLDER_MIN_SIZE

def only_once(slug: str, side: str, holders: list) -> bool:
    ret = False
    for holder in holders:
        if is_good(holder):
            print(f"{holder['name']} {holder['proxyWallet']} {holder['amount']:.2f} shares {side} {holder['traded_count']} trades")
        key = f"{slug}-{side}-{holder['proxyWallet']}"
        # print(f"key: {key}")
        if once.get(key):
            print(f"{key} in once")
        if is_good(holder) and not once.get(key):
            ret = True
            once[key] = True
    return ret

def get_size(holders: list) -> float:
    size = 0
    for holder in holders:
        size += float(holder["amount"])
    return size

def get_sifu_size(holders: list) -> float:
    size = 0
    for holder in holders:
        if holder['proxyWallet'] in sifu_addresses:
            size += float(holder["amount"])
    return size

def get_last_minute_size(holders: list) -> float:
    size = 0
    for holder in holders:
        if holder['proxyWallet'] in WHITELIST:
            size += float(holder["amount"]) - float(holder["last_amount"])
    return size

def print_holders(interval: int, symbol: str):
    global sifu_size, last_sifu_size
    now = int(time.time())
    seconds_left = 900 - (now % 900)

    market = clob_api.get_market(interval, symbol)

    holders_by_side = clob_api.get_holders(market)
    for side in holders_by_side:
        holders = holders_by_side[side]
        slug = f"{symbol}-updown-15m-{interval}"

        last_sifu_size = sifu_size
        sifu_size = get_sifu_size(holders)
        total_size = get_size(holders)

        if total_size < MIN_TOTAL_SIZE and (DEBUG or not only_once(slug, side, holders)) and abs(sifu_size - last_sifu_size) < 0.00000001:
            print(f"Skipping {slug} {side} total {total_size:.2f} shares sifu {sifu_size:.2f} ({sifu_size - last_sifu_size:+.2f}) shares")
            continue
        
        num_wallets = 0
        num_shares = 0
        message = ""
        for holder in holders:
            if not is_good(holder):
                continue
            num_wallets += 1
            num_shares += float(holder["amount"])
            

        # price = bid if side == "Up" else (1 - ask)
        message = "\n".join([
            f"sifu {100*sifu_size / total_size:.2f}% = {sifu_size:.2f} ({sifu_size - last_sifu_size:+.2f}) / {total_size:.2f} shares {side}",
            f"<a href=\"https://polymarket.com/event/{slug}\">{slug}</a>",
            "",
        ])
        for holder in holders:
            if not is_good(holder):
                continue

            sifu = 'sifu ' if holder['proxyWallet'] in sifu_addresses else ''

            text = f"{sifu}<a href=\"https://polymarket.com/profile/{holder['proxyWallet']}\">{holder['name']}</a> {holder['amount']:.2f} shares {side}\n"
            message += text

        key = f"{slug}-{side}"
        if (total_size > MIN_TOTAL_SIZE and (DEBUG or not once.get(key, False))):
            once[key] = True
            telegram.send_message(message, disable_web_page_preview=True)
        else:
            print(message)


def run():
    interval = int(time.time() // 900 * 900)
    for symbol in SYMBOLS:
        print_holders(interval, symbol)

def main():
    run()
    while True:
        time.sleep(1)
        now = int(time.time()) + 10
        if (now % 60) == 0:
            try:
                run()
            except Exception as e:
                print(f"Error: {e}")
                time.sleep(30)


if __name__ == "__main__":
    main()
