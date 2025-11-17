import csv
from datetime import datetime, timezone

from poly_market_maker.price_engine import PriceEngine

import logging
from prometheus_client import start_http_server
import os  
import time

from poly_market_maker.price_engine import PriceEngine
from poly_market_maker.price_feed import PriceFeedClob
from poly_market_maker.my_token import MyToken, Collateral


from poly_market_maker.utils import setup_logging, setup_web3
from poly_market_maker.market import Market
from poly_market_maker.clob_api import ClobApi

from dotenv import load_dotenv          # Environment variable management
load_dotenv()                           # Load environment variables from .env file

FUNDER = os.getenv("FUNDER")
TARGET = os.getenv("TARGET")
DEBUG = True

setup_logging()
logger = logging.getLogger(__name__)

client = ClobApi()

def main():
    engine = PriceEngine(symbol="btc/usd")
    engine.start()

    header = ["timestamp", "price", "bid", "ask"]

    interval = None
    price_feed = None

    try:
        while True:
            time.sleep(1)
            now = int(time.time())
            if interval != now // 900:  # 15-min intervals
                interval = now // 900
                market = client.get_market(interval * 900) 
                price_feed = PriceFeedClob(market, client)
                engine.target_price = engine.price

            price = engine.get_price()
            timestamp = engine.get_timestamp()

            if price is not None and timestamp is not None:
                bid, ask = price_feed.get_bid_ask(MyToken.A)
                row = [int(timestamp), price, bid, ask]

                date = datetime.fromtimestamp(timestamp, tz=timezone.utc).strftime("%Y-%m-%d")

                path = f"./data/price_{date}.csv"
                file_exists = os.path.exists(path)

                write_header = False
                if not file_exists:
                    write_header = True

                with open(path, mode='a', newline='') as file:
                    writer = csv.writer(file)
                    if write_header:
                        writer.writerow(header)
                        write_header = False

                    writer.writerow(row)
           

    except KeyboardInterrupt:
        print("Stopping engine...")
        engine.stop()
        print("Stopped.")

if __name__ == "__main__":
    if DEBUG:
        main()
    else:
        while True:
            try:
                main()
            except Exception as e:
                print("Error occurred, restarting:", e)
                continue