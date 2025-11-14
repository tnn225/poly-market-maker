import websocket
import threading
import json
import time
from datetime import datetime, timezone
from collections import deque
import numpy as np
from scipy.stats import norm
import logging

logging.basicConfig(level=logging.INFO)


class PriceEngine(threading.Thread):
    def __init__(self, symbol, target_price=None, interval_seconds=900, vol_window=10000):
        super().__init__(daemon=True)

        self.WS_URL = "wss://ws-live-data.polymarket.com"

        # Config
        self.symbol = symbol
        self.interval_seconds = interval_seconds
        self.vol_window = vol_window

        # State
        self.price = None
        self.target_price = target_price
        now_ts = time.time()  # current time in seconds
        self.interval_start = datetime.fromtimestamp(now_ts - (now_ts % self.interval_seconds), tz=timezone.utc)
        self.prices = deque(maxlen=vol_window)

        # Thread safety
        self.lock = threading.Lock()
        self._stop_event = threading.Event()

        # WebSocket object
        self.ws = None

    # ==========================================================
    #               VOLATILITY  & PROBABILITY
    # ==========================================================

    def estimate_sigma(self, price_list):
        # Not enough points to compute log returns
        if len(price_list) < 3:
            return 0.0

        log_returns = np.diff(np.log(price_list))

        # Additional protection: if std cannot be computed (NaN)
        sigma = np.std(log_returns, ddof=1)
        if not np.isfinite(sigma):
            return 0.0

        return sigma

    def prob_up(self, P_open, P_now, seconds_left, sigma, mu=0):
        if not P_open or not P_now:
            return 1 
        
        if seconds_left <= 0 or sigma == 0:
            return float(P_now >= P_open)

        z = (
            np.log(P_now / P_open)
            - (mu - 0.5 * sigma**2) * seconds_left
        ) / (sigma * np.sqrt(seconds_left))

        return float(norm.cdf(z))

    # ==========================================================
    #                WEBSOCKET  CALLBACKS
    # ==========================================================

    def on_open(self, ws):
        logging.info("Connected to Polymarket RTDS")

        subscribe = {
            "action": "subscribe",
            "subscriptions": [{
                "topic": "crypto_prices_chainlink",
                "type": "*",
                "filters": ""
            }]
        }

        ws.send(json.dumps(subscribe))
        logging.info(f"Subscribed to {self.symbol}")

    def on_message(self, ws, message):
        if not message.strip():
            return

        try:
            data = json.loads(message)
        except:
            return

        payload = data.get("payload")
        if not payload:
            return

        # Match correct symbol
        if payload.get("symbol") != self.symbol:
            return

        # timestamp is ms
        ts = payload["timestamp"] / 1000
        price = payload["value"]
        now = datetime.fromtimestamp(ts, tz=timezone.utc)

        with self.lock:
            self.price = price
            self.prices.append(price)
            if self.target_price is None:
                self.target_price = price
            # New interval?
            if int(now.timestamp()) % self.interval_seconds == 0:
                self.interval_start = now
                self.target_price = price
                logging.info(f"🕐 New interval at {self.interval_start}, target={self.target_price:.2f}")

    def on_error(self, ws, error):
        logging.error(f"WebSocket error: {error}")

    def on_close(self, ws, code, msg):
        logging.warning(f"WebSocket closed {code} {msg}")

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
                logging.error(f"WS crashed: {e}")
                time.sleep(2)

    def stop(self):
        self._stop_event.set()
        if self.ws:
            self.ws.close()

    # ==========================================================
    #                     EXTERNAL API
    # ==========================================================

    def get_data(self):
        with self.lock:
            if self.interval_start is None or self.price is None:
                return None

            now = datetime.now(timezone.utc)
            elapsed = (now - self.interval_start).total_seconds()
            seconds_left = max(0, self.interval_seconds - elapsed)
            sigma = self.estimate_sigma(list(self.prices))
            p = self.prob_up(self.target_price, self.price, seconds_left, sigma)

            delta = self.price - self.target_price
            delta_pct = delta / self.target_price

            return {
                "timestamp": round(now.timestamp()),
                "symbol": self.symbol,
                "price": round(self.price, 3),
                "target_price": round(self.target_price, 3),
                "seconds_left": round(seconds_left),
                "sigma": round(sigma, 6),
                "prob_up": round(p, 3),
                "delta": round(delta, 3),
                "delta_pct": round(delta_pct, 4),
            }


# ==========================================================
#                HOW TO USE THE ENGINE
# ==========================================================

if __name__ == "__main__":
    engine = PriceEngine(symbol="btc/usd", interval_seconds=900)
    engine.start()

    print("🚀 PriceEngine started...")

    try:
        while True:
            state = engine.get_state()
            time.sleep(1)

    except KeyboardInterrupt:
        print("Stopping engine...")
        engine.stop()
        print("Stopped.")
