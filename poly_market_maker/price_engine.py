import csv
import os
import websocket
import threading
import json
import time
from datetime import datetime, timezone
import logging
from poly_market_maker.intervals import Interval

logging.basicConfig(level=logging.INFO)

class PriceEngine(threading.Thread):
    def __init__(self, symbol: str):
        super().__init__(daemon=True)

        self.WS_URL = "wss://ws-live-data.polymarket.com"
        self.symbol = symbol

        # Shared state
        self.price = None
        self.timestamp = None

        intervals = Interval()
        self.target = None
        while self.target is None:
            time.sleep(1)
            timestamp = int(time.time())
            timestamp = timestamp // 900 * 900
            data = intervals.get_data('BTC', timestamp)
            print(f"Data: {data}")
            if data is not None:
                self.interval = data['interval']
                self.target = data['openPrice']
                print(f"Target: {self.target}")
        
        self.interval = 0
        date = datetime.fromtimestamp(int(time.time()), tz=timezone.utc).strftime("%Y-%m-%d")
        filename = f'./data/price_{date}.csv'
        if os.path.exists(filename):
            self.read_prices(filename)

        # Thread safety
        self.lock = threading.Lock()
        self._stop_event = threading.Event()
        self.ws = None

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

            if self.timestamp % 900 == 0:
                self.interval = self.timestamp
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
            return {
                'timestamp': self.timestamp,
                'price': self.price,
                'target': self.target,
                'interval': self.interval
            }

# ==========================================================
#                 HOW TO USE THE ENGINE
# ==========================================================

if __name__ == "__main__":
    price_engine = PriceEngine(symbol="btc/usd")
    price_engine.start()

    print("🚀 PriceEngine started...")

    try:
        while True:
            timestamp = price_engine.get_timestamp()
            price = price_engine.get_price()
            interval = price_engine.get_interval()
            target = price_engine.get_target()
            print(f"timestamp {timestamp} price {price} target {target}")
            time.sleep(1)

    except KeyboardInterrupt:
        print("Stopping engine...")
        price_engine.stop()
        print("Stopped.")
