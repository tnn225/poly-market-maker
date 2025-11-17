import websocket
import json
import threading
import time
import logging
from queue import Queue

logging.basicConfig(level=logging.INFO)

class BidAskEngine(threading.Thread):
    def __init__(self, symbol, write_queue=None, clob_auth=None, gamma_auth=None):
        super().__init__(daemon=True)
        self.symbol = symbol.lower()
        self.ws_url = "wss://ws-live-data.polymarket.com"
        self.write_queue = write_queue
        self.clob_auth = clob_auth
        self.gamma_auth = gamma_auth

        self.price = None
        self.timestamp = None
        self.lock = threading.Lock()
        self._stop_event = threading.Event()
        self.ws = None

    def on_open(self, ws):
        logging.info(f"Connected to Polymarket RTDS for {self.symbol}")
        subscribe_msg = {
            "action": "subscribe",
            "subscriptions": [
                {
                    "topic": "crypto_prices_chainlink",
                    "type": "*",
                    "filters": json.dumps({"symbol": self.symbol}),
                    "clob_auth": self.clob_auth,
                    "gamma_auth": self.gamma_auth
                }
            ]
        }
        ws.send(json.dumps(subscribe_msg))

    def on_message(self, ws, message):
        try:
            msg = json.loads(message)
        except json.JSONDecodeError:
            return

        payload = msg.get("payload")
        if not payload:
            return

        timestamp = payload.get("timestamp") or msg.get("timestamp")
        price = payload.get("value")

        if price is not None:
            with self.lock:
                self.price = price
                self.timestamp = timestamp
                if self.write_queue:
                    self.write_queue.put({
                        "symbol": self.symbol,
                        "price": price,
                        "timestamp": timestamp
                    })

    def on_error(self, ws, error):
        logging.error(f"RTDS WS error: {error}")

    def on_close(self, ws, code, msg):
        logging.warning(f"RTDS WS closed {code} {msg}")

    def run(self):
        while not self._stop_event.is_set():
            try:
                self.ws = websocket.WebSocketApp(
                    self.ws_url,
                    on_open=self.on_open,
                    on_message=self.on_message,
                    on_error=self.on_error,
                    on_close=self.on_close,
                )
                self.ws.run_forever(ping_interval=5, ping_timeout=2)
            except Exception as e:
                logging.error(f"WS crashed: {e}")
                time.sleep(1)

    def stop(self):
        self._stop_event.set()
        if self.ws:
            self.ws.close()

    def get_price(self):
        with self.lock:
            return self.price
