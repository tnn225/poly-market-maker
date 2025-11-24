import csv
import datetime
import os
import time
from poly_market_maker.clob_api import ClobApi
from py_clob_client.clob_types import TradeParams

from poly_market_maker.my_token import MyToken

# ---------------- CONFIG ----------------
ADDRESS = "0x5248313731287b61d714ab9df655442d6ed28aa2"
IS_MAKER = False    # True = maker trades, False = taker trades
# ----------------------------------------

client = ClobApi()

import requests
import csv
import time

BASE_URL = "https://data-api.polymarket.com/trades"
# Example filters

headers = ['timestamp', 'proxyWallet', 'side', 'price', 'size', 'asset']

def fetch_trades(condition_ids: list, token_1):
    limit = 500
    offset = 0
    trades = []

    while True:
        params = {
            "limit": limit,
            "offset": offset,
            "market": ",".join(condition_ids),
            "takerOnly": False,
            "user": ADDRESS
        }
        resp = requests.get(BASE_URL, params=params)
        print(f"Fetching trades with params: {params}: {resp}")
        resp.raise_for_status()

        if resp.status_code != 200:
            print(f"Error fetching trades: {resp.status_code}")
            return []

        data = resp.json()
        print(f"Data: {len(data)}")

        for data in data:
            trade = {}
            for header in headers:
                trade[header] = data.get(header, None)
            
            trade['size'] = round(float(trade['size']), 2)
            trade['price'] = round(float(trade['price']),2)

            trades.append(trade)
            if trade['side'] == 'SELL':
                trade['size'] = -float(trade['size'])

            if str(trade['asset']) == str(token_1):
                trade['token 1'] = round(trade['size'], 2)
                trade['price 1'] = round(trade['price'], 2)

                trade['token 2'] = 0
                trade['price 2'] = round(1 - trade['price 1'], 2)
            else:
                trade['token 2'] = round(trade['size'], 2)
                trade['price 2'] = round(trade['price'], 2)

                trade['token 1'] = 0
                trade['price 1'] = round(1 - trade['price 2'], 2)

        if len(data) < limit:
            break
        offset += len(data)

    return trades

def save_trades(interval: int, trades: list):
    headers = ['timestamp', 'proxyWallet', 'token 1', 'price 1', 'token 2', 'price 2']
    trades = sorted(trades, key=lambda x: int(x['timestamp']))

    # File path
    path = f"./data/trades_{interval}.csv"
 
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
    for i in range(100):
        interval = (now // 900 - i - 1) * 900
        market = client.get_market(interval) 

        trades = fetch_trades([str(market.condition_id)], market.token_id(MyToken.A))
        save_trades(interval, trades)
        print(len(trades), "trades found.")
    # for trade in trades:
    #     print(trade)

if __name__ == "__main__":
    main()
