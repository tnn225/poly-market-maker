import os
import time
from datetime import datetime, timezone
from typing import Dict, Optional

from dotenv import load_dotenv
load_dotenv()

from poly_market_maker.constants import DEBUG
import requests

from poly_market_maker.utils.telegram import Telegram
from poly_market_maker.utils.common import format_address
telegram = Telegram()


INFO_URL = "https://api.hyperliquid.xyz/info"
POLL_SECONDS = 60 if not DEBUG else 5

WATCH_WALLETS = [
    x.strip().lower()
    for x in os.getenv("WATCH_WALLETS", "").split(",")
    if x.strip()
]


def get_usdc_balance(address: str) -> float:
    payload = {"type": "spotClearinghouseState", "user": address.lower()}
    r = requests.post(INFO_URL, json=payload, timeout=10)
    r.raise_for_status()
    data = r.json()
    for bal in data.get("balances", []):
        if bal["coin"] == "USDC":
            return float(bal["total"])
    return 0.0


def get_total_raw_usd(address: str) -> float:
    """Return totalRawUsd from clearinghouseState marginSummary."""
    payload = {"type": "clearinghouseState", "user": address.lower()}
    r = requests.post(INFO_URL, json=payload, timeout=10)
    r.raise_for_status()
    data = r.json()
    summary = data.get("marginSummary") or {}
    raw = summary.get("totalRawUsd", "0")
    return float(raw)


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def main() -> None:
    last: Dict[str, Optional[float]] = {}

    # Initialize baseline (so it doesn't print immediately)
    for a in WATCH_WALLETS:
        addr = a.lower()
        try:
            last[addr] = get_total_raw_usd(addr)
        except Exception as e:
            print(f"[{now_iso()}] init error {addr}: {e}")
            last[addr] = None

    print(f"Watching {len(WATCH_WALLETS)} addresses for USDC balance changes (every {POLL_SECONDS}s).")
    print("Ctrl+C to stop.\n")

    while True:
        loop_start = time.time()

        message = ""

        for a in WATCH_WALLETS:
            addr = a.lower()
            try:
                cur = get_total_raw_usd(addr)
                print(f"{addr}: ${cur:.2f}")
                prev = last.get(addr)

                # If we previously failed (prev is None), set baseline quietly
                if prev is None:
                    last[addr] = cur
                    continue

                if cur != prev or DEBUG:
                    message += f"<a href=\"https://hypurrscan.io/address/{addr}#txs\">{format_address(addr)}</a> : ${prev:.2f} -> ${cur:.2f}\n"
                    last[addr] = cur
            except Exception as e:
                print(f"erro    r {addr}: {e}")
        if message:
            telegram.send_message(message)
        # Sleep remaining time
        elapsed = time.time() - loop_start
        time.sleep(max(0, POLL_SECONDS - elapsed))


if __name__ == "__main__":
    main()
