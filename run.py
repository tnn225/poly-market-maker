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


setup_logging()
logger = logging.getLogger(__name__)



if __name__ == "__main__":
    engine = PriceEngine(symbol="btc/usd", interval_seconds=900)
    engine.start()

    client = ClobApi()
    slug = f"btc-updown-15m-{int(engine.interval_start.timestamp())}"
    condition_id = client.get_condition_id_by_slug(slug)

    market = Market(
        condition_id,
        client.get_collateral_address(),
    )

    price_feed = PriceFeedClob(market, client)

    print("🚀 PriceEngine started...")
    header = ["timestamp", "symbol", "price", "delta", "delta_pct", "sigma", "seconds_left", "prob_up", "bid", "ask"]

    try:
        while True:
            data = engine.get_data()
            # print(data)
            # print(market.token_ids)
            bid, ask = price_feed.get_bid_ask(MyToken.A)
            # print(bid, ask)

            if data:
                row = [data["timestamp"], data["symbol"], data["price"], data["delta"], data["delta_pct"], data["sigma"], data["seconds_left"], data["prob_up"], bid, ask]
                print(f"BTC={row[2]:.2f} | Δ={row[3]:+.2f} ({row[4]:+.3f}% | σ={row[5]:.6f} | sec_left={int(row[6]):4d} | ProbUp={row[7]:5.3f} | Bid={row[8]} | Ask={row[9]}")

                date = datetime.fromtimestamp(data["timestamp"], tz=timezone.utc).strftime("%Y-%m-%d")

                path = f"./data/price_data_{date}.csv"
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

            # print(data["symbol"], data["price"], data["delta"], data["delta_pct"], data["sigma"], data["seconds_left"], data["prob_up"], bid, ask)
            time.sleep(1)

    except KeyboardInterrupt:
        print("Stopping engine...")
        engine.stop()
        print("Stopped.")


    