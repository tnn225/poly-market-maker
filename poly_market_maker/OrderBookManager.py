import asyncio                      # Asynchronous I/O
import json                        # JSON handling
import websockets                  # WebSocket client
import traceback                   # Exception handling
import time

from sortedcontainers import SortedDict

from poly_market_maker.clob_api import ClobApi

def process_book_data(token, json_data):
    # Convert to list of tuples (price, size) as floats
    bids = [(float(entry['price']), float(entry['size'])) for entry in json_data['bids']]
    asks = [(float(entry['price']), float(entry['size'])) for entry in json_data['asks']]

    # Sort bids descending (highest bid first), asks ascending (lowest ask first)
    bids.sort(key=lambda x: x[0], reverse=True)
    asks.sort(key=lambda x: x[0])
    best_bid = bids[0][0] if len(bids) else None
    best_ask = asks[0][0] if len(asks) else None
    print(token, best_bid, best_ask)
    return best_bid, best_ask

def process_price_change(token, best_bid, best_ask):
    print(token, best_bid, best_ask)
    pass

def process_data(json_datas):
    # print("Processing data...", json_datas)
    if not isinstance(json_datas, list):
        json_datas = [json_datas]

    for json_data in json_datas:
        # print("json_data", json_data)
        event_type = json_data['event_type']
        asset = json_data['market']

        if event_type == 'book':
            token = json_data.get('asset_id', None)    
            process_book_data(token, json_data)
                
        elif event_type == 'price_change':
            for data in json_data['price_changes']:
                token = data.get('asset_id', None)
                best_bid = float(data['best_bid'])
                best_ask = float(data['best_ask'])

                process_price_change(token, best_bid, best_ask)

async def connect_market_websocket(chunk):
    """
    Connect to Polymarket's market WebSocket API and process market updates.
    
    This function:
    1. Establishes a WebSocket connection to the Polymarket API
    2. Subscribes to updates for a specified list of market tokens
    3. Processes incoming order book and price updates
    
    Args:
        chunk (list): List of token IDs to subscribe to
        
    Notes:
        If the connection is lost, the function will exit and the main loop will
        attempt to reconnect after a short delay.
    """
    uri = "wss://ws-subscriptions-clob.polymarket.com/ws/market"
    async with websockets.connect(uri, ping_interval=5, ping_timeout=None) as websocket:
        # Prepare and send subscription message
        message = {"assets_ids": chunk}
        await websocket.send(json.dumps(message))

        print("\n")
        print(f"Sent market subscription message: {message}")

        try:
            # Process incoming market data indefinitely
            while True:
                message = await websocket.recv()
                json_data = json.loads(message)
                # Process order book updates and trigger trading as needed
                process_data(json_data)
        except websockets.ConnectionClosed:
            print("Connection closed in market websocket")
            print(traceback.format_exc())
        except Exception as e:
            print(f"Exception in market websocket: {e}")
            print(traceback.format_exc())
        finally:
            # Brief delay before attempting to reconnect
            await asyncio.sleep(5)

if __name__ == "__main__":
    client = ClobApi()
    now = int(time.time())
    interval = now // 900
    market = client.get_market(interval * 900) 
    token_ids = [str(token) for token in market.token_ids.values()]

    asyncio.run(connect_market_websocket(token_ids))

