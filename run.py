import csv
from datetime import datetime, timezone
import logging
import os  
import time

from poly_market_maker.order_book_engine import OrderBookEngine
from poly_market_maker.price_engine import PriceEngine
from poly_market_maker.price_feed import PriceFeedClob
from poly_market_maker.my_token import MyToken
from poly_market_maker.utils import setup_logging
from poly_market_maker.clob_api import ClobApi
from poly_market_maker.prediction_engine import PredictionEngine

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
prediction_engine = PredictionEngine()
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
 
    while True:
        time.sleep(0.1)

        timestamp = engine.get_timestamp()
        price = engine.get_price()
        data = engine.get_data()
        target = data.get('target')
        
        if timestamp is None or timestamp == last:
            continue
        last = timestamp
 
        now = int(timestamp)
        seconds_left = 900 - (now % 900)
        if interval == None or now // 900 > interval:  # 15-min intervals
            interval = now // 900
            market = client.get_market(interval * 900) 
            if 'order_book' in locals():
                order_book.stop()
            order_book = OrderBookEngine(market)
            order_book.start()

        bid, ask = order_book.get_bid_ask(MyToken.A)
        bid_b, ask_b = order_book.get_bid_ask(MyToken.B)
        # print(f"Bid B: {bid_b}, Ask B: {ask_b}")
        
        # Calculate probability if target is available
        up = None
        if target is not None and price is not None:
            up = prediction_engine.get_probability(price, target, seconds_left)
        
        row = [int(timestamp), price, bid, ask]
        write_row(row)
        
        if target is not None and up is not None:
            delta = price - target
            print(f"{seconds_left} {price} {delta:.4f} Bid: {bid}, Ask: {ask}, Up: {up:.3f}")
        else:
            print(f"{seconds_left} {price} Bid: {bid}, Ask: {ask}")

if __name__ == "__main__":
    main()
