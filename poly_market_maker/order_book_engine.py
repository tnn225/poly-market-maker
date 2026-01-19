from collections import defaultdict
import websocket
import threading
import json
import time
import logging

from poly_market_maker.clob_api import ClobApi
from poly_market_maker.my_token import MyToken

logging.basicConfig(level=logging.INFO)

class OrderBookEngine(threading.Thread):
    def __init__(self, market):
        super().__init__(daemon=True)

        self.WS_URL = "wss://ws-subscriptions-clob.polymarket.com/ws/market"
        self.market = market

        # shared state
        self.best_bid = defaultdict(float)
        self.best_ask = defaultdict(float)

        # threading & safety
        self.lock = threading.Lock()
        self._stop_event = threading.Event()
        self.ws = None

    # ==========================================================
    #                WEBSOCKET CALLBACKS
    # ==========================================================

    def on_open(self, ws):
        logging.info("Connected to Polymarket CLOB WS")
        self.asset_ids = [str(token) for token in self.market.token_ids.values()]
        ws.send(json.dumps({"assets_ids": self.asset_ids}))
        logging.info(f"Subscribed to {len(self.asset_ids)} assets")
        # Start ping thread if needed
        # threading.Thread(target=self.ping, args=(ws,), daemon=True).start()

    def on_message(self, ws, message):
        if not message or message in ("PING", "PONG"):
            return

        try:
            data_list = json.loads(message)
        except json.JSONDecodeError:
            return

        if not isinstance(data_list, list):
            data_list = [data_list]

        for data in data_list:
            event_type = data.get("event_type")
            if event_type == "book":
                token = int(data.get("asset_id"))
                self.process_book_data(token, data)

            elif event_type == "price_change":
                for pc in data.get("price_changes", []):
                    token = int(pc.get("asset_id"))
                    bid = float(pc.get("best_bid", 0)) if pc.get("best_bid") else None
                    ask = float(pc.get("best_ask", 0)) if pc.get("best_ask") else None
                    self.process_price_change(token, bid, ask)

    def on_error(self, ws, error):
        logging.error(f"WebSocket error: {error}")

    def on_close(self, ws, code, msg):
        logging.warning(f"WebSocket closed: code={code}, msg={msg}")

    # ==========================================================
    #                ORDERBOOK PROCESSING
    # ==========================================================

    def process_book_data(self, token, data):
        bids = [(float(e["price"]), float(e["size"])) for e in data.get("bids", [])]
        asks = [(float(e["price"]), float(e["size"])) for e in data.get("asks", [])]

        bids.sort(key=lambda x: x[0], reverse=True)
        asks.sort(key=lambda x: x[0])

        with self.lock:
            self.best_bid[token] = bids[0][0] if bids else None
            self.best_ask[token] = asks[0][0] if asks else None

    def process_price_change(self, token, bid, ask):
        with self.lock:
            self.best_bid[token] = bid
            self.best_ask[token] = ask

    # ==========================================================
    #                  THREAD LOOP
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
    #                ORDERBOOK GETTERS
    # ==========================================================

    def get_bid_ask(self, token: MyToken):
        token_id = self.market.token_id(token)
        with self.lock:
            return self.best_bid[token_id], self.best_ask[token_id]

# ==========================================================
#               HOW TO USE THE ENGINE
# ==========================================================

if __name__ == "__main__":
    client = ClobApi()
    now = int(time.time())
    interval = now // 900
    market = client.get_market(interval * 900)

    order_book_engine = OrderBookEngine(market) 
    order_book_engine.start()

    while True:
        now = int(time.time())
        if now // 900 > interval:
            interval = now // 900
            order_book_engine.stop()
            market = client.get_market(interval * 900)
            order_book_engine = OrderBookEngine(market) 
            order_book_engine.start()

        bid, ask = order_book_engine.get_bid_ask(MyToken.A)
        seconds_left = 900 - (now % 900)
        print(f"{seconds_left}s bid: {bid}, ask: {ask}")
        time.sleep(5)
