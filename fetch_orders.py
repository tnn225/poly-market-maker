import requests
import csv
import time

# ============ NHẬP THÔNG TIN TẠI ĐÂY ============
USER = "0x38973f5b3abbbcbed16ee15c2baa5a3f16b843ab" # Wacha LTF
USER = "0x6031b6eed1c97e853c6e0f03ad3ce3529351f96d" # Gabagool22
# USER = "0x589222a5124a96765443b97a3498d89ffd824ad2" # PurpleThunderBicycleMountain
# USER = "0xa5e83423126dbc6cdb34f10f37f5d27668ab95f5" # haidcks116
USER = "0x2e33c2571dcca96cd8e558dcf8195c738b82d046" # MiyooMarketMaker
USER = "0x6ce61e4481377d8ee002be65f9dad04683ac2ab9" # tsafwan39
USER = "0x69d2648cf20d9d57c6ba5319aa09dda43bd522fb" # unknow
AUTH_TOKEN = "b3471958-6e00-47a1-a5b8-f3eb7e144edb"
MAX_RETRIES = 5
# ================================================



def fetch_with_retry(url: str, headers: dict, params: dict, max_retries: int = 5, base_delay: float = 1.0):
    """Fetch URL with exponential backoff retry."""
    for attempt in range(max_retries):
        try:
            response = requests.get(url, headers=headers, params=params, timeout=30)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            if attempt == max_retries - 1:
                print(f"Failed after {max_retries} attempts: {e}")
                raise

            delay = base_delay * (2 ** attempt)
            print(f"Attempt {attempt + 1} failed: {e}. Retrying in {delay}s...")
            time.sleep(delay)

    return None


def fetch_all_orders(user: str, market_slug: str, auth_token: str, max_retries: int = 5) -> list:
    """Fetch all orders from the API with pagination and retry support."""
    base_url = "https://api.domeapi.io/v1/polymarket/orders"
    headers = {"Authorization": f'Bearer {auth_token}'}
    # print(headers)
    all_orders = []
    offset = 0
    limit = 1000

    while True:
        params = {
            "limit": limit,
            "offset": offset,
            "user": user,
            "market_slug": market_slug
        }

        data = fetch_with_retry(base_url, headers, params, max_retries=max_retries)

        if data is None:
            print("Failed to fetch data, stopping.")
            break

        orders = data.get("orders", [])
        all_orders.extend(orders)

        pagination = data.get("pagination", {})
        print(f"Fetched {len(orders)} orders (offset: {offset}, total: {pagination.get('total', 'N/A')})")

        if not pagination.get("has_more", False):
            break

        offset += limit

    for order in all_orders:
        order["timestamp"] = int(order["timestamp"]) - TIMESTAMP
        order["shares_normalized"] = round(order["shares_normalized"], 2)
        order["price"] = round(order["price"], 2)
        
        # Transform: if buying Down, convert to selling Up
        # if order.get('token_label') == 'Down' and order.get('side') == 'BUY':
        #    order['token_label'] = 'Up'
        #    order['side'] = 'SELL'
        #    order['price'] = round(1 - order['price'], 2)

    all_orders.sort(key=lambda x: x["timestamp"])

    return all_orders


def save_to_csv(orders: list, filename: str):
    """Save orders to CSV file."""
    if not orders:
        # print("No orders to save.")
        return

    fieldnames = [
         "timestamp", "token_label", "side", "price", "shares_normalized"
    ]

    with open(filename, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(orders)

    print(f"Saved {len(orders)} orders to {filename}")


if __name__ == "__main__":
    now = int(time.time())
    TIMESTAMP = now - (now % 900) if False else 1770891300
    TIMESTAMP = 1770907500

    for symbol in ["eth", "btc", "sol", "xrp"]:
        MARKET_SLUG = f"{symbol}-updown-15m-{TIMESTAMP}"
        OUTPUT_FILE = f"./data/orders/{TIMESTAMP}_{USER}.csv"

        print(f"\nFetching orders for user: {USER} slug: {MARKET_SLUG}")
        orders = fetch_all_orders(USER, MARKET_SLUG, AUTH_TOKEN, MAX_RETRIES)
        save_to_csv(orders, f"./data/orders/{symbol}_{TIMESTAMP}_{USER}.csv")