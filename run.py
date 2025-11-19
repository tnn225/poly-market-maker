import csv
from datetime import datetime, timezone

from poly_market_maker.order_book_engine import OrderBookEngine
from poly_market_maker.price_engine import PriceEngine

import logging
from prometheus_client import start_http_server
import os  
import time

from poly_market_maker.price_engine import PriceEngine
from poly_market_maker.price_feed import PriceFeedClob
from poly_market_maker.my_token import MyToken, Collateral


from poly_market_maker.price_prediction import PricePrediction
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
engine = PriceEngine(symbol="btc/usd")
engine.start()
header = ["timestamp", "price", "bid", "ask"]


# Ensure ./data exists
os.makedirs("./data", exist_ok=True)

def write_row(row):
    timestamp = row[0]

    # Convert timestamp → yyyy-mm-dd
    date = datetime.fromtimestamp(timestamp, tz=timezone.utc).strftime("%Y-%m-%d")

    # File path
    path = f"./data/price_{date}.csv"
    file_exists = os.path.exists(path)

    with open(path, mode="a", newline="") as file:
        writer = csv.writer(file)

        # Write header only once
        if not file_exists:
            writer.writerow(header)

        # Write the row
        writer.writerow(row)

def get_action(bid, ask, up, down, spread):
    action = ""
    if bid and up and bid > 0:
        value = (up - bid) / bid 
        if value > spread:
            action = "BUY"
    if ask and down and ask < .99:
        value = (down - (0.99 - ask)) / (1 - ask)
        if value > spread: 
            action = "SELL"
    return action

def main():
    interval = None 
    last = None

    now = int(time.time())
    timestamp = now // 900 * 900 
    prediction = PricePrediction(timestamp)

    spread = 0.05

    while True:
        time.sleep(0.1)

        timestamp = engine.get_timestamp()
        price = engine.get_price()
        # target = price
        # 
        if timestamp is None or timestamp == last:
            continue
        last = timestamp
        prediction.add_price(timestamp, price)

        now = int(timestamp)
        seconds_left = 900 - (now % 900)
        if interval != now // 900:  # 15-min intervals
            interval = now // 900
            market = client.get_market(interval * 900) 
            token_id = market.token_id(MyToken.A)
            bid, ask = client.get_bid_ask(token_id)
            order_book = OrderBookEngine(market, token_id, bid, ask)
            order_book.start()

        bid, ask = order_book.get_bid_ask(MyToken.A)
        bid_b, ask_b = order_book.get_bid_ask(MyToken.B)
        print(f"Bid B: {bid_b}, Ask B: {ask_b}")
        row = [int(timestamp), price, bid, ask]
        write_row(row)

        up = prediction.get_probability(price, seconds_left)
        if up is None:
            print(f"{seconds_left}s {price} Bid: {bid}, Ask: {ask}")
            continue
        down = 1 - up 

        action = get_action(bid, ask, up, down, spread)

        print(f"{seconds_left}s {price} {price - prediction.target:+6.2f}: Bid: {bid}, Ask: {ask} Up {up:.4f} Down {down:.4f} {action}")

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

