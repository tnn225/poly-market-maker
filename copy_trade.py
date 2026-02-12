import logging
import threading
import time

from poly_market_maker.utils import setup_logging
from poly_market_maker.strategies.copy_trade_strategy import CopyTradeStrategy


setup_logging()
logger = logging.getLogger(__name__)

def main():
    duration = 15
    symbols = ["eth", "btc", "sol", "xrp"] if duration == 15 else ["btc"]
    maker = None
    maker_thread = None
    while True:
        now = int(time.time()) + 10
        interval = now // 900 * 900
        if maker is None or interval != maker.start_time:
            for symbol in symbols:
                maker = CopyTradeStrategy(interval=interval, symbol=symbol, duration=duration)
                maker_thread = threading.Thread(target=maker.run, daemon=True)
                maker_thread.start()
        time.sleep(1)

if __name__ == "__main__":
    main()
