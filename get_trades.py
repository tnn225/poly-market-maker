import csv
import datetime
import os
import time
from poly_market_maker.clob_api import ClobApi
from py_clob_client.clob_types import TradeParams

from poly_market_maker.my_token import MyToken

# ---------------- CONFIG ----------------
ADDRESS = "0x6031b6eed1c97e853c6e0f03ad3ce3529351f96d"
IS_MAKER = False    # True = maker trades, False = taker trades
# ----------------------------------------

client = ClobApi()

import requests
import csv
import time

BASE_URL = "https://data-api.polymarket.com/trades"
# Example filters

headers = ['timestamp', 'proxyWallet', 'side', 'price', 'size', 'asset']

def fetch_trades(condition_id: list, token_1):
    limit = 500
    offset = 0
    trades = []

    while True:
        params = {
            "limit": limit,
            "offset": offset,
            "market": condition_id,
            # "user": user
        }
        resp = requests.get(BASE_URL, params=params)
        print(f"Fetching trades with params: {params}: {resp}")
        resp.raise_for_status()
        for data in resp.json():
            trade = {}
            for header in headers:
                trade[header] = data.get(header, None)
            trades.append(trade)
            if trade['side'] == 'SELL':
                trade['size'] = -float(trade['size'])

            if trade['asset'] == str(token_1):
                trade['token 1'] = float(trade['size'])
                trade['price 1'] = float(trade['price'])

                trade['token 2'] = 0
                trade['price 2'] = 1 - trade['price 1']
            else:
                trade['token 2'] = float(trade['size'])
                trade['price 2'] = float(trade['price'])

                trade['token 1'] = 0
                trade['price 1'] = 1 - trade['price 2']
 
        if len(resp.json()) < limit:
            break
        offset += len(resp.json())

    return trades

def save_trades(filename: str, trades: list):
    headers = ['timestamp', 'proxyWallet', 'token 1', 'price 1', 'token 2', 'price 2']

    # File path
    path = f"./data/trades_{filename}.csv"
 
    with open(path, mode="w", newline="") as file:
        writer = csv.writer(file)

        writer.writerow(headers)

        for trade in trades:
            row = []
            for header in headers:
                row.append(trade.get(header, None))
            writer.writerow(row)

def main():
    now = int(time.time())
    interval = now // 900 - 100
    market = client.get_market(interval * 900) 

    trades = fetch_trades(market.condition_id, market.token_id(MyToken.A))
    save_trades(market.condition_id, trades)
    print(len(trades), "trades found.")
    # for trade in trades:
    #     print(trade)

if __name__ == "__main__":
    main()
