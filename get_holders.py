import time

from poly_market_maker.utils import setup_logging
from poly_market_maker.clob_api import ClobApi

setup_logging()

clob_api = ClobApi()

def main():
    while True:
        time.sleep(1)
        now = int(time.time()) + 10
        if (now % 900) == 0:
            interval = int(time.time() // 900 * 900)
            clob_api.print_holders(interval)

if __name__ == "__main__":
    main()
