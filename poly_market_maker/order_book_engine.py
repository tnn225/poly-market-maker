from websocket import WebSocketApp
import json
import time
import threading

from poly_market_maker.clob_api import ClobApi

MARKET_CHANNEL = "market"

class OrderBookEngine(threading.Thread):

    def __init__(self, market, token_id, bid, ask):
        super().__init__(daemon=True)

        # store token ids
        self.data = [str(token) for token in market.token_ids.values()]

        # orderbook data
        self.best_bid = {}
        self.best_ask = {}
        self.best_bid[token_id] = bid
        self.best_ask[token_id] = ask

        # lock for thread safety
        self.lock = threading.Lock()

        # websocket
        self.ws = WebSocketApp(
            "wss://ws-subscriptions-clob.polymarket.com/ws/market",
            on_message=self.on_message,
            on_error=self.on_error,
            on_close=self.on_close,
            on_open=self.on_open,
        )

    # ------------------------------------------------------------
    # WebSocket callbacks
    # ------------------------------------------------------------

    def on_message(self, ws, message):
        # print("Received message:", message)

        # skip heartbeat or empty messages
        if not message or message == "PONG" or message == "PING":
            return

        # Try parse JSON
        try:
            json_datas = json.loads(message)
        except json.JSONDecodeError:
            # ignore any non-JSON message
            return

        # print("Processing data...", json_datas)
        if not isinstance(json_datas, list):
            json_datas = [json_datas]

        for json_data in json_datas:
            # print("json_data", json_data)
            event_type = json_data['event_type']
            asset = json_data['market']
            # print("Event type:", event_type, "Asset:", asset)

            if event_type == 'book':
                token = int(json_data.get('asset_id', None))
                self.process_book_data(token, json_data)
                    
            elif event_type == 'price_change':
                for data in json_data['price_changes']:
                    token = int(data.get('asset_id', None))
                    best_bid = float(data['best_bid'])
                    best_ask = float(data['best_ask'])

                    self.process_price_change(token, best_bid, best_ask)

    def on_error(self, ws, error):
        print("Error:", error)

    def on_close(self, ws, code, msg):
        print("WebSocket closed:", code, msg)

    def on_open(self, ws):
        print("WebSocket opened")
        ws.send(json.dumps({"assets_ids": self.data}))

        # start ping thread
        ping_thread = threading.Thread(target=self.ping, args=(ws,), daemon=True)
        ping_thread.start()

    # ------------------------------------------------------------
    # Threads
    # ------------------------------------------------------------

    def ping(self, ws):
        while True:
            try:
                ws.send("PING")
            except:
                return
            time.sleep(10)

    def run(self):
        self.ws.run_forever()

    # ------------------------------------------------------------
    # Orderbook processing
    # ------------------------------------------------------------

    def process_book_data(self, token, json_data):
        bids = [(float(e['price']), float(e['size'])) for e in json_data.get('bids', [])]
        asks = [(float(e['price']), float(e['size'])) for e in json_data.get('asks', [])]

        bids.sort(key=lambda x: x[0], reverse=True)
        asks.sort(key=lambda x: x[0])

        best_bid = bids[0][0] if bids else None
        best_ask = asks[0][0] if asks else None

        with self.lock:
            self.best_bid[token] = best_bid
            self.best_ask[token] = best_ask

        # print(f"[OB] {token} | bid={best_bid}, ask={best_ask}")
        return best_bid, best_ask

    def process_price_change(self, token, best_bid, best_ask):
        with self.lock:
            self.best_bid[token] = best_bid
            self.best_ask[token] = best_ask

    def get_bid_ask(self, token):
        with self.lock:
            return self.best_bid.get(token), self.best_ask.get(token)


# ------------------------------------------------------------
# Main entry
# ------------------------------------------------------------

if __name__ == "__main__":
    client = ClobApi()

    now = int(time.time())
    interval = now // 900
    market = client.get_market(interval * 900)

    market_connection = OrderBookEngine(market)
    market_connection.start()       # <-- Start WebSocket thread

    # Example: run your market-making loop
    while True:
        for token in market.token_ids.values():
            bid, ask = market_connection.get_bid_ask(str(token))
            print(">>>", token, "| bid:", bid, "ask:", ask)
        time.sleep(2)


