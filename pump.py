import asyncio
import websockets
import json
import requests
from datetime import datetime

# ─── Config ───────────────────────────────────────────────────────────────────
WS_URL = "wss://pumpportal.fun/api/data"   # Free, no API key required
TOTAL_SUPPLY = 1_000_000_000               # pump.fun tokens always have 1B supply

graduated_today = []

# ─── Fetch price + market cap from DexScreener ────────────────────────────────
def get_token_info(mint: str) -> dict:
    """
    Queries DexScreener for price and market cap after graduation.
    Falls back to pump.fun's own API if DexScreener has no data yet.
    """
    try:
        # DexScreener - usually has data within seconds of graduation
        url = f"https://api.dexscreener.com/latest/dex/tokens/{mint}"
        resp = requests.get(url, timeout=5)
        data = resp.json()
        pairs = data.get("pairs") or []

        if pairs:
            # Pick the most liquid pair
            pair = sorted(pairs, key=lambda p: float(p.get("liquidity", {}).get("usd", 0) or 0), reverse=True)[0]
            price_usd   = float(pair.get("priceUsd") or 0)
            market_cap  = float(pair.get("marketCap") or 0) or (price_usd * TOTAL_SUPPLY)
            liquidity   = float((pair.get("liquidity") or {}).get("usd") or 0)
            volume_24h  = float((pair.get("volume") or {}).get("h24") or 0)
            return {
                "price_usd":  price_usd,
                "market_cap": market_cap,
                "liquidity":  liquidity,
                "volume_24h": volume_24h,
                "source":     "dexscreener"
            }
    except Exception:
        pass

    # Fallback: pump.fun internal API
    try:
        url = f"https://frontend-api.pump.fun/coins/{mint}"
        resp = requests.get(url, timeout=5)
        data = resp.json()
        usd_market_cap = data.get("usd_market_cap", 0)
        # Rough price from market cap
        price_usd = usd_market_cap / TOTAL_SUPPLY if usd_market_cap else 0
        return {
            "price_usd":  price_usd,
            "market_cap": usd_market_cap,
            "liquidity":  0,
            "volume_24h": 0,
            "source":     "pump.fun"
        }
    except Exception:
        pass

    return {"price_usd": 0, "market_cap": 0, "liquidity": 0, "volume_24h": 0, "source": "unavailable"}


# ─── Print a graduation event ─────────────────────────────────────────────────
def print_graduation(msg: dict, info: dict):
    name   = msg.get("name", "Unknown")
    symbol = msg.get("symbol", "?")
    mint   = msg.get("mint", "")
    now    = datetime.now().strftime("%H:%M:%S")

    mc  = info["market_cap"]
    px  = info["price_usd"]
    liq = info["liquidity"]
    vol = info["volume_24h"]

    mc_str  = f"${mc:>12,.0f}" if mc  else "   (fetching...)"
    px_str  = f"${px:.8f}"     if px  else "N/A"
    liq_str = f"${liq:>10,.0f}" if liq else "N/A"
    vol_str = f"${vol:>10,.0f}" if vol else "N/A"

    print(f"\n{'─'*60}")
    print(f"🎓  GRADUATED  [{now}]")
    print(f"    Name      : {name} (${symbol})")
    print(f"    Mint      : {mint}")
    print(f"    Price USD : {px_str}")
    print(f"    Mkt Cap   : {mc_str}")
    print(f"    Liquidity : {liq_str}")
    print(f"    Vol 24h   : {vol_str}")
    print(f"    Source    : {info['source']}")
    print(f"    Link      : https://pump.fun/{mint}")


# ─── Handle a single migration message ────────────────────────────────────────
async def handle_migration(msg: dict):
    mint = msg.get("mint")
    if not mint:
        return

    # Brief wait — DexScreener needs a moment to index the new pool
    await asyncio.sleep(3)

    info = await asyncio.to_thread(get_token_info, mint)
    print_graduation(msg, info)

    graduated_today.append({
        "name":       msg.get("name"),
        "symbol":     msg.get("symbol"),
        "mint":       mint,
        "time":       datetime.now().isoformat(),
        "price_usd":  info["price_usd"],
        "market_cap": info["market_cap"],
        "liquidity":  info["liquidity"],
    })


# ─── Main WebSocket listener ──────────────────────────────────────────────────
async def listen():
    print("Connecting to PumpPortal WebSocket...")
    print("Listening for graduation events. Press Ctrl+C to stop.\n")

    while True:  # auto-reconnect loop
        try:
            async with websockets.connect(WS_URL, ping_interval=30) as ws:
                # Subscribe to migration (graduation) events
                await ws.send(json.dumps({"method": "subscribeMigration"}))
                print("✅ Subscribed to graduation/migration events\n")

                async for raw in ws:
                    msg = json.loads(raw)

                    # PumpPortal sends a txType field on migration events
                    tx_type = msg.get("txType", "")
                    if tx_type in ("migrate", "migration") or msg.get("mint"):
                        asyncio.create_task(handle_migration(msg))

        except websockets.exceptions.ConnectionClosed:
            print("⚠️  Connection closed. Reconnecting in 5s...")
            await asyncio.sleep(5)
        except Exception as e:
            print(f"⚠️  Error: {e}. Reconnecting in 5s...")
            await asyncio.sleep(5)


# ─── Entry point ─────────────────────────────────────────────────────────────
async def main():
    try:
        await listen()
    except KeyboardInterrupt:
        print(f"\n\n📊 Total graduations captured: {len(graduated_today)}")
        if graduated_today:
            with open("graduated_tokens.json", "w") as f:
                json.dump(graduated_today, f, indent=2)
            print("💾 Saved to graduated_tokens.json")

if __name__ == "__main__":
    asyncio.run(main())