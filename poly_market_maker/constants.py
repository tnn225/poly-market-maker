import os

from dotenv import load_dotenv
load_dotenv()

OK = "OK"
MIN_TICK = 0.01
MIN_PRICE = 1.0
MIN_SIZE = 10
MAX_DECIMALS = 2
DEBUG = False
# DEBUG = True


# get_holders thresholds (relaxed when DEBUG)
HOLDER_MIN_SIZE = 1000 if DEBUG else 10000
MIN_TOTAL_SIZE = 1000 if DEBUG else 60000
MAX_TRADED_COUNT = 1000 if DEBUG else 30


def _env_address_list(key: str) -> list:
    raw = os.getenv(key, "")
    return list(dict.fromkeys(x.strip().lower() for x in raw.split(",") if x.strip()))


WHITELIST = _env_address_list("WHITELIST")
BLACKLIST = _env_address_list("BLACKLIST")
BRAIN_WALLETS = _env_address_list("BRAIN_WALLETS")
GOAT_WALLETS = _env_address_list("GOAT_WALLETS")


def in_whitelist(wallet: str) -> bool:
    return wallet.lower() in WHITELIST
