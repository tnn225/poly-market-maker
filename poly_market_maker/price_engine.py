import csv
import os
import websocket
import threading
import json
import time
from datetime import datetime, timezone
import logging

from poly_market_maker.prediction_engine import PredictionEngine

logging.basicConfig(level=logging.INFO)

class PriceEngine(threading.Thread):
    _instances: dict[str, "PriceEngine"] = {}

    def __new__(cls, symbol: str, *args, **kwargs):
        key = (symbol or "").strip().lower()
        inst = cls._instances.get(key)
        if inst is None:
            inst = super().__new__(cls)
            cls._instances[key] = inst
            inst._singleton_key = key
            inst._initialized = False
        return inst

    def __init__(self, symbol: str):
        # Avoid re-initializing the same per-symbol singleton instance
        if getattr(self, "_initialized", False):
            return

        super().__init__(daemon=True)

        self.prediction_engine = PredictionEngine()

        self.WS_URL = "wss://ws-live-data.polymarket.com"
        self.symbol = (symbol or "").strip().lower()

        # Shared state
        self.price = None
        self.timestamp = None
        self.target = None
        self.interval = None
        self.prob_est = None
        self.sigma = None

        date = datetime.fromtimestamp(int(time.time()), tz=timezone.utc).strftime("%Y-%m-%d")
        filename = f'./data/prices/price_{date}.csv'
        if os.path.exists(filename):
            print(f"Reading prices from {filename}")
            self.read_prices(filename)

        # Thread safety
        self.lock = threading.Lock()
        self._stop_event = threading.Event()
        self.ws = None

        self._initialized = True
        self.start()

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
                self.prediction_engine.add_price(price)
                if timestamp % 900 == 0:

                    self.interval = timestamp
                    self.target = price
                    print(f"Read target {self.target} for interval {self.interval}")

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
            self.prediction_engine.add_price(value)
            current_interval = int(self.timestamp) // 900 * 900
            if self.interval != current_interval:
                self.interval = current_interval
                # Best-effort: if we missed the exact open, use first observed tick.
                self.target = self.price
            elif int(self.timestamp) % 900 == 0:
                self.target = self.price


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

    def get_sigma(self):
        return self.prediction_engine.get_sigma()

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
            if self.timestamp is None or self.price is None or self.target is None:
                return None
            seconds_left = 900 - (int(self.timestamp) % 900)
            up = self.prediction_engine.get_probability(self.price, self.target, seconds_left)
            return {
                'timestamp': self.timestamp,
                'price': self.price,
                'target': self.target,  
                'interval': self.interval,
                'up': up,
            }
            return data

# ==========================================================
#                 HOW TO USE THE ENGINE
# ==========================================================

if __name__ == "__main__":
    price_engine = PriceEngine(symbol="btc/usd")

    print("🚀 PriceEngine started...")

    try:
        while True:
            data = price_engine.get_data()

            if data and data['timestamp'] is not None and data['price'] is not None and data['target'] is not None  :
                price = data.get('price')
                target = data.get('target')
                delta = price - target
                seconds_left = 900 - (int(data['timestamp']) % 900)
                up = data.get("up")
                if up is None:
                    print(f"{seconds_left} {price:.2f} {delta:+.2f} up: None")
                else:
                    print(f"{seconds_left} {price:.2f} {delta:+.2f} up: {up:.2f}")
            else:
                print("No data")
            time.sleep(1)

    except KeyboardInterrupt:
        print("Stopping engine...")
        price_engine.stop()
        print("Stopped.")
