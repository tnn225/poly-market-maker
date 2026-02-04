import time

from poly_market_maker.utils import setup_logging
from poly_market_maker.clob_api import ClobApi
from poly_market_maker.utils.telegram import Telegram
from poly_market_maker.my_token import MyToken
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


 # engine = PriceEngine(symbol="btc/usd")

SIDES = {MyToken.A: "Up", MyToken.B: "Down"}
SYMBOLS = ["btc"] # , "eth", "sol", "xrp"]

once = {}


def in_brain(wallet: str) -> bool:
    return wallet.lower() in BRAIN_WALLETS


def is_whitelisted(holders: list) -> bool:
    for holder in holders:
        wallet = holder["proxyWallet"].lower()
        if wallet in WHITELIST and abs(holder["amount"] - holder["last_amount"]) > 0.0001:
            return True
    return False

def is_good(holder: dict) -> bool:
    if holder["proxyWallet"].lower() in WHITELIST:
        return True
    if holder["proxyWallet"].lower() in BLACKLIST:
        return False
    if holder['traded_count'] > 1000: 
        return False
    if holder['amount'] < 1000:
        return False
    return holder['amount'] > HOLDER_MIN_SIZE or holder['traded_count'] < MAX_TRADED_COUNT

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

def get_whitelisted_size(holders: list) -> float:
    size = 0
    for holder in holders:
        if holder['proxyWallet'] in WHITELIST:
            size += float(holder["amount"]) 
    return size

def get_last_minute_size(holders: list) -> float:
    size = 0
    for holder in holders:
        if holder['proxyWallet'] in WHITELIST:
            size += float(holder["amount"]) - float(holder["last_amount"])
    return size

def print_holders(interval: int, symbol: str):
    now = int(time.time())
    seconds_left = 900 - (now % 900)

    market = clob_api.get_market(interval, symbol)

    holders_by_side = clob_api.get_holders(market)
    for side in holders_by_side:
        holders = holders_by_side[side]
        slug = f"{symbol}-updown-15m-{interval}"

        total_size = get_size(holders)

        print(f"\nPrinting holders for interval: {interval} {symbol} {side} {total_size:.2f}")

        if total_size < MIN_TOTAL_SIZE and (DEBUG or not only_once(slug, side, holders)) and not is_whitelisted(holders):
            print(f"Skipping {slug} {side} {total_size:.2f}")
            continue
        
        #delta = engine.get_delta()
        # bid, ask = 0.5, 0.5
        """
        try:
            bid, ask = clob_api.get_bid_ask(market)
        except Exception as e:
            print(f"Error getting bid ask: {e}")
            bid, ask = 0.5, 0.5

        if bid is None or ask is None or bid <= 0.02 or ask >= 0.98:
            print(f"Skipping {slug} {side} {total_size:.2f} bid: {bid} ask: {ask}")
            continue
        """
        
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
            f"{num_wallets} fresh wallets with {100*num_shares / total_size:.2f}% = {num_shares:.2f} / {total_size:.2f} shares {side}",
            f"<a href=\"https://polymarket.com/event/{slug}\">{slug}</a>",
            "",
            f"Sifu: {get_whitelisted_size(holders):.2f} ({get_last_minute_size(holders):+.2f}) shares {side}",
            # f"Delta: {'+' if (d := engine.get_delta()) >= 0 else '-'}${abs(d):.2f}",
            "",
            "",
        ])
        for holder in holders:
            if not is_good(holder):
                continue

            sifu = '🦶 ' if in_whitelist(holder["proxyWallet"]) else '💩'
            if in_brain(holder["proxyWallet"]):
                sifu = '❤️ '
            if holder["proxyWallet"].lower() in GOAT_WALLETS:
                sifu = '🐐 '
            last = holder.get("last_amount", holder["amount"])
            avg_price = holder.get("avg_price", 0.0)
            cost = holder.get("cost", 0.0)
            text = f"{sifu}<a href=\"https://polymarket.com/profile/{holder['proxyWallet']}\">{holder['name']}</a> {holder['amount']:.2f} ({holder['amount'] - last:+.2f}) shares {side} at ${avg_price:.2f} = ${cost:.2f}\n"
            message += text

        if (total_size > MIN_TOTAL_SIZE and (DEBUG or not once.get(slug, False))) or is_whitelisted(holders):
            once[slug] = True
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
        if (now % 20) == 0:
            try:
                run()
            except Exception as e:
                print(f"Error: {e}")
                time.sleep(30)


if __name__ == "__main__":
    main()
