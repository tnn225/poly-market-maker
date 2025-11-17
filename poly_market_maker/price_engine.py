import websocket
import threading
import json
import time
from datetime import datetime, timezone
import logging

logging.basicConfig(level=logging.INFO)

class PriceEngine(threading.Thread):
    def __init__(self, symbol: str):
        super().__init__(daemon=True)

        self.WS_URL = "wss://ws-live-data.polymarket.com"
        self.symbol = symbol

        # Shared state
        self.price = None
        self.timestamp = None

        # Thread safety
        self.lock = threading.Lock()
        self._stop_event = threading.Event()
        self.ws = None

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
            self.timestamp = ts / 1000
            self.price = value

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


# ==========================================================
#                 HOW TO USE THE ENGINE
# ==========================================================

if __name__ == "__main__":
    engine = PriceEngine(symbol="btc/usd")
    engine.start()

    print("🚀 PriceEngine started...")

    try:
        while True:
            print("Current Price:", engine.get_price())
            time.sleep(1)

    except KeyboardInterrupt:
        print("Stopping engine...")
        engine.stop()
        print("Stopped.")
