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

class BinanceEngine(threading.Thread):
    def __init__(self, symbol: str):
        super().__init__(daemon=True)

        # Binance websocket base URL
        self.WS_URL = "wss://stream.binance.com:9443/ws"
        self.symbol = symbol
        self._binance_symbol = self._normalize_binance_symbol(symbol)

        # Shared state
        self.price = None
        self.timestamp = None
        self.target = None
        self.interval = None

        self._max_df_rows = 10_000
        self._rolling_sigma_window = 300

        dataset = Dataset(days=1)
        self.df = dataset.df[
            [
                "timestamp",
                "price",
                "target",
                "interval",
                "seconds_left",
                "log_return",
                "dt",
                "log_return_step",
                "log_return_per_sec",
                "sigma",
                "sigma_annual",
                "z_score",
                "prob_est",
            ]
        ].copy()
        if len(self.df) > self._max_df_rows:
            self.df = self.df.iloc[-self._max_df_rows :].reset_index(drop=True)

        if "sigma_annual" not in self.df.columns and "sigma" in self.df.columns:
            seconds_per_year = 365.0 * 24.0 * 60.0 * 60.0
            self.df["sigma_annual"] = self.df["sigma"] * np.sqrt(seconds_per_year)

        if not self.df.empty:
            last = self.df.iloc[-1]
            self.interval = int(last["interval"])
            self.target = float(last["target"])

        date = datetime.fromtimestamp(int(time.time()), tz=timezone.utc).strftime("%Y-%m-%d")
        filename = f'./data/prices/price_{date}.csv'
        if os.path.exists(filename):
            self.read_prices(filename)

        # Thread safety
        self.lock = threading.Lock()
        self._stop_event = threading.Event()
        self.ws = None

    @staticmethod
    def _normalize_binance_symbol(symbol: str) -> str:
        """
        Convert common symbol formats (e.g. 'btc/usd', 'BTC-USD', 'btcusdt') to
        a Binance stream symbol (lowercase, no separators).

        Note: Binance uses USDT for most USD-quoted pairs (BTCUSDT, ETHUSDT, ...).
        """
        s = (symbol or "").strip().lower()
        s = s.replace("/", "").replace("-", "").replace("_", "")
        if s.endswith("usd") and not s.endswith("usdt"):
            s = s[:-3] + "usdt"
        return s

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

        # Keep at most one row per second (Binance can emit multiple trades/second)
        if not self.df.empty and int(self.df["timestamp"].iloc[-1]) == int(self.timestamp):
            idx = self.df.index[-1]
            self.df.at[idx, "price"] = price
            self.df.at[idx, "target"] = target_f
            self.df.at[idx, "interval"] = int(interval)
            self.df.at[idx, "seconds_left"] = seconds_left
            self.df.at[idx, "log_return"] = log_return
        else:
            self.df.loc[len(self.df)] = {
                "timestamp": int(self.timestamp),
                "price": price,
                "target": target_f,
                "interval": int(interval),
                "seconds_left": seconds_left,
                "log_return": log_return,
                "dt": np.nan,
                "log_return_step": np.nan,
                "log_return_per_sec": np.nan,
                "sigma": np.nan,
                "sigma_annual": np.nan,
                "z_score": np.nan,
                "prob_est": np.nan,
            }

        # Keep memory bounded
        if len(self.df) > self._max_df_rows:
            self.df = self.df.iloc[-self._max_df_rows :].reset_index(drop=True)

        # Per-second vol (match Dataset):
        #   log_return_step = log(price_t / price_{t-1})
        #   log_return_per_sec = log_return_step / dt_seconds
        if len(self.df) >= 2:
            dt = float(max(1, int(self.df["timestamp"].iloc[-1]) - int(self.df["timestamp"].iloc[-2])))
            prev_price = float(self.df["price"].iloc[-2])
            step = float(np.log(price / prev_price)) if prev_price > 0 else 0.0
        else:
            dt = 1.0
            step = 0.0

        self.df.at[self.df.index[-1], "dt"] = dt
        self.df.at[self.df.index[-1], "log_return_step"] = step
        self.df.at[self.df.index[-1], "log_return_per_sec"] = (step / dt) if dt > 0 else 0.0

        # Time-based rolling window over last N seconds
        roll_s = int(max(1, int(self._rolling_sigma_window)))
        ts_index = pd.to_datetime(self.df["timestamp"], unit="s")
        sigma_s = (
            self.df.set_index(ts_index)["log_return_per_sec"]
            .astype(float)
            .rolling(f"{roll_s}s", min_periods=2)
            .std()
        )
        self.df["sigma"] = sigma_s.to_numpy()
        self.df["sigma"] = self.df["sigma"].fillna(0.0)

        seconds_per_year = 365.0 * 24.0 * 60.0 * 60.0
        self.df["sigma_annual"] = self.df["sigma"] * np.sqrt(seconds_per_year)

        # z_score using annualized volatility:
        #   sigma_annual = sigma_per_sec * sqrt(seconds_per_year)
        # Remaining-time vol = sigma_annual * sqrt(seconds_left / seconds_per_year)
        seconds_per_year = 365.0 * 24.0 * 60.0 * 60.0
        time_factor = np.sqrt(self.df["seconds_left"].astype(float) / seconds_per_year)
        sigma_scaled = self.df["sigma_annual"].replace(0, np.nan)
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
        logging.info(f"Connected to Binance stream: {self._binance_symbol}@trade")

    

    def on_message(self, ws, message):
        if not message.strip():
            return

        try:
            data = json.loads(message)
        except json.JSONDecodeError:
            return

        # Binance trade stream payload
        # Example fields:
        #  e=trade, E=eventTime(ms), s=symbol, p=price(str), T=tradeTime(ms)
        ts_ms = data.get("T") or data.get("E")
        price_str = data.get("p")
        if ts_ms is None or price_str is None:
            return
        try:
            ts_sec = int(int(ts_ms) / 1000)
            px = float(price_str)
        except Exception:
            return

        with self.lock:
            self.timestamp = ts_sec
            self.price = px
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
                url = f"{self.WS_URL}/{self._binance_symbol}@trade"
                self.ws = websocket.WebSocketApp(
                    url,
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

    def get_price_at(self, timestamp: int) -> float | None:
        """
        Return the last known price at or before `timestamp` (seconds).
        Uses the engine's rolling dataframe (one row per second).
        """
        ts = int(timestamp)
        with self.lock:
            if self.df.empty:
                return None
            # Ensure timestamps are sorted increasing (they should be)
            arr = self.df["timestamp"].to_numpy(dtype=np.int64)
            idx = int(np.searchsorted(arr, ts, side="right") - 1)
            if idx < 0:
                return None
            return float(self.df["price"].iloc[idx])

# ==========================================================
#                 HOW TO USE THE ENGINE
# ==========================================================

if __name__ == "__main__":
    binance_engine = BinanceEngine(symbol="btc/usdt")
    binance_engine.start()

    print("🚀 BinanceEngine started...")

    try:
        while True:
            data = binance_engine.get_data()
            if data and data['timestamp'] is not None and data['price'] is not None and data['target'] is not None  :
                seconds_left = 900 - (int(data['timestamp']) % 900)
                print(f"{seconds_left} {data['price']:.2f} {data['delta']:+.2f} up: {data['prob_est']:.2f}")
            else:
                print("No data")
            time.sleep(1)

    except KeyboardInterrupt:
        print("Stopping engine...")
        binance_engine.stop()
        print("Stopped.")
