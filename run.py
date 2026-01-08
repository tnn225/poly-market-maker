import csv
from datetime import datetime, timezone
import logging
import os  
import time

from poly_market_maker.order_book_engine import OrderBookEngine
from poly_market_maker.price_engine import PriceEngine
from poly_market_maker.my_token import MyToken
from poly_market_maker.clob_api import ClobApi

from dotenv import load_dotenv          # Environment variable management
load_dotenv()                           # Load environment variables from .env file


client = ClobApi()
engine = PriceEngine(symbol="btc/usd")
engine.start()
header = ["timestamp", "price", "bid", "ask"]

# Ensure ./data exists
os.makedirs("./data/prices", exist_ok=True)

def write_row(row):
    timestamp = row[0]

    # Convert timestamp → yyyy-mm-dd
    date = datetime.fromtimestamp(timestamp, tz=timezone.utc).strftime("%Y-%m-%d")

    # File path
    path = f"./data/prices/price_{date}.csv"
    file_exists = os.path.exists(path)

    with open(path, mode="a", newline="") as file:
        writer = csv.writer(file)

        # Write header only once
        if not file_exists:
            writer.writerow(header)

        # Write the row
        writer.writerow(row)


def main():
    interval = None 
    last = None

    now = int(time.time())
    timestamp = now // 900 * 900 
 
    while True:
        time.sleep(0.1)

        data = engine.get_data()
        target = data.get('target')
        timestamp = data.get('timestamp')
        price = data.get('price')

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
        
        row = [int(timestamp), price, bid, ask]
        write_row(row)
        
        if target is not None:
            delta = price - target
            print(f"{seconds_left} {price} {delta:+.4f} Bid: {bid}, Ask: {ask}")
        else:
            print(f"{seconds_left} {price} Bid: {bid}, Ask: {ask}")

if __name__ == "__main__":
    while True:
        try:
            main()
        except Exception as e:
            print(f"Error: {e}")
            time.sleep(60)
