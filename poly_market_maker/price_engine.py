import csv
import os
import websocket
import threading
import json
import time
from datetime import datetime, timezone
import logging
from poly_market_maker.intervals import Interval
import pandas as pd
import numpy as np
from scipy.stats import norm

from poly_market_maker.dataset import Dataset

logging.basicConfig(level=logging.INFO)

class PriceEngine(threading.Thread):
    def __init__(self, symbol: str):
        super().__init__(daemon=True)

        self.WS_URL = "wss://ws-live-data.polymarket.com"
        self.symbol = symbol

        # Shared state
        self.price = None
        self.timestamp = None
        self.target = None
        self.interval = None


        dataset = Dataset(days=1)
        self.df = dataset.df[['timestamp', 'price', 'target', 'interval', 'seconds_left', 'log_return', 'sigma', 'z_score', 'prob_est']]

        self._max_df_rows = 10_000
        self._rolling_sigma_window = 60

        date = datetime.fromtimestamp(int(time.time()), tz=timezone.utc).strftime("%Y-%m-%d")
        filename = f'./data/prices/price_{date}.csv'
        if os.path.exists(filename):
            self.read_prices(filename)

        # Thread safety
        self.lock = threading.Lock()
        self._stop_event = threading.Event()
        self.ws = None

    def _append_prob_est_row_locked(self):
        """
        Append latest tick to self.df and compute prob_est.

        Must be called with self.lock held.
        """
        if self.timestamp is None or self.price is None:
            return

        interval = self.interval
        target = self.target

        if interval is None:
            interval = int(self.timestamp) // 900 * 900
            self.interval = interval

        # If we don't have a target for this interval yet, we can't compute prob_est.
        if target is None:
            return

        seconds_left = 900 - (int(self.timestamp) - int(interval))
        seconds_left = max(1, min(900, int(seconds_left)))

        price = float(self.price)
        target_f = float(target)
        if price <= 0 or target_f <= 0:
            return

        log_return = float(np.log(price / target_f))

        self.df.loc[len(self.df)] = {
            "timestamp": int(self.timestamp),
            "price": price,
            "target": target_f,
            "interval": int(interval),
            "seconds_left": seconds_left,
            "log_return": log_return,
            "sigma": np.nan,
            "z_score": np.nan,
            "prob_est": np.nan,
        }

        # Keep memory bounded
        if len(self.df) > self._max_df_rows:
            self.df = self.df.iloc[-self._max_df_rows :].reset_index(drop=True)

        # Rolling sigma on log_return (same spirit as Dataset)
        self.df["sigma"] = self.df["log_return"].rolling(
            window=self._rolling_sigma_window, min_periods=1
        ).std()

        # z_score scales volatility by time remaining
        time_factor = np.sqrt(self.df["seconds_left"].astype(float) / 60.0)
        sigma_scaled = self.df["sigma"].replace(0, np.nan)
        self.df["z_score"] = self.df["log_return"] / (sigma_scaled * time_factor)
        self.df["z_score"] = self.df["z_score"].fillna(0.0)
        self.df["prob_est"] = norm.cdf(self.df["z_score"].astype(float))

    def read_prices(self, filename):
        with open(filename, mode='r', newline='') as file:
            csv_reader = csv.reader(file)

            # Optionally, skip the header row if present
            header = next(csv_reader)
            print(f"Header: {header}")

            # Iterate and print each data row
            for row in csv_reader:
                # print(f"Data Row: {row}")
                timestamp = int(row[0])
                price = float(row[1])
                if timestamp % 900 == 0:
                    self.interval = timestamp
                    self.target = price
                    # print(f"Read target {self.target} for interval {self.interval}")

    # ==========================================================
    #                WEBSOCKET CALLBACKS
    # ==========================================================

    def on_open(self, ws):
        logging.info("Connected to Polymarket RTDS")

        subscribe = {
            "action": "subscribe",
            "subscriptions": [{
                "topic": "crypto_prices_chainlink",
                "type": "*",
                "filters": f"{{\"symbol\":\"{self.symbol}\"}}"
            }]
        }

        ws.send(json.dumps(subscribe))
        logging.info(f"Subscribed to symbol: {self.symbol}")

    

    def on_message(self, ws, message):
        if not message.strip():
            return

        try:
            data = json.loads(message)
        except json.JSONDecodeError:
            return

        payload = data.get("payload")
        if not payload:
            return

        # symbol filter
        if payload.get("symbol") != self.symbol:
            return

        # Some messages DO NOT have timestamp/value
        ts = payload.get("timestamp")
        value = payload.get("value")

        if ts is None or value is None:
            # Ignore incomplete payloads
            return

        with self.lock:
            self.timestamp = round(ts / 1000)
            self.price = value
            current_interval = int(self.timestamp) // 900 * 900
            if self.interval != current_interval:
                self.interval = current_interval
                # Best-effort: if we missed the exact open, use first observed tick.
                self.target = self.price
            elif int(self.timestamp) % 900 == 0:
                self.target = self.price

            self._append_prob_est_row_locked()

    def on_error(self, ws, error):
        logging.error(f"WebSocket error: {error}")

    def on_close(self, ws, code, msg):
        logging.warning(f"WebSocket closed: code={code}, msg={msg}")

    # ==========================================================
    #                     THREAD LOOP
    # ==========================================================

    def run(self):
        while not self._stop_event.is_set():
            try:
                self.ws = websocket.WebSocketApp(
                    self.WS_URL,
                    on_open=self.on_open,
                    on_message=self.on_message,
                    on_error=self.on_error,
                    on_close=self.on_close,
                )
                self.ws.run_forever()
            except Exception as e:
                logging.error(f"WebSocket crashed: {e}")
                time.sleep(2)

    def stop(self):
        self._stop_event.set()
        if self.ws:
            try:
                self.ws.close()
            except:
                pass

    # ==========================================================
    #                    EXTERNAL GETTERS
    # ==========================================================

    def get_price(self):
        with self.lock:
            return self.price

    def get_timestamp(self):
        with self.lock:
            return self.timestamp
        
    def get_interval(self):
        with self.lock:
            return self.interval
        
    def get_target(self):
        with self.lock:
            return self.target
        
    def get_data(self):
        with self.lock:
            prob_est = None
            if not self.df.empty and "prob_est" in self.df.columns:
                prob_est = float(self.df["prob_est"].iloc[-1])
            return {
                'timestamp': self.timestamp,
                'price': self.price,
                'target': self.target,
                'delta': self.price - self.target if self.target and self.price else None,
                'interval': self.interval,
                'prob_est': prob_est,
            }

    def get_df(self) -> pd.DataFrame:
        with self.lock:
            return self.df.copy()

# ==========================================================
#                 HOW TO USE THE ENGINE
# ==========================================================

if __name__ == "__main__":
    price_engine = PriceEngine(symbol="btc/usd")
    price_engine.start()

    print("🚀 PriceEngine started...")

    try:
        while True:
            data = price_engine.get_data()
            if data and data['timestamp'] is not None and data['price'] is not None and data['target'] is not None  :
                seconds_left = 900 - (int(data['timestamp']) % 900)
                print(f"{seconds_left} {data['price']:.2f} {data['delta']:+.2f} up: {data['prob_est']:.2f}")
            else:
                print("No data")
            time.sleep(1)

    except KeyboardInterrupt:
        print("Stopping engine...")
        price_engine.stop()
        print("Stopped.")
