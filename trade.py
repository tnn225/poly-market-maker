import logging
import time

from poly_market_maker.utils import setup_logging
from poly_market_maker.strategies.pair_strategy import PairStrategy
from poly_market_maker.strategies.balance_strategy import BalanceStrategy

setup_logging()
logger = logging.getLogger(__name__)

def main():
    maker = None
    while True:
        now = int(time.time()) + 10
        interval = now // 900 * 900
        if maker is None or interval != maker.start_time:
            maker = BalanceStrategy(interval=interval)
            maker.run()
        time.sleep(1)

if __name__ == "__main__":
    main()
