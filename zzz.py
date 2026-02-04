import requests

INFO_URL = "https://api.hyperliquid.xyz/info"


def get_data(address: str) -> int:
    payload = {
        "type": "clearinghouseState",
        "user": address.lower()
    }

    r = requests.post(INFO_URL, json=payload, timeout=10)
    r.raise_for_status()
    data = r.json()
    print(f"{address}: {data}")

    return data


addr = "0xcb5da67d3e80cca90ed16096b97f97e4bb4ace2e"
print("Nonce:", get_nonce(addr))
