"""
Fetch trending pump.fun tokens from DexScreener (pumpswap), ordered by liquidity.
For each mint, fetch top holders with token balance and SOL balance.
"""
import os
import requests
from dotenv import load_dotenv

load_dotenv()
HELIUS_KEY = os.getenv("HELIUS_KEY")
RPC = f"https://rpc.helius.xyz/?api-key={HELIUS_KEY}"

DEXSCREENER_URL = "https://api.dexscreener.com/latest/dex/search"
DEX_ID = "pumpswap"


def get_top_holders(mint: str, rpc_url: str = RPC, limit: int = 20) -> list[dict]:
    """
    Fetch top token holders with token balance and SOL balance.
    Returns list of {owner, tokens, sol}.
    """
    resp = requests.post(rpc_url, json={
        "jsonrpc": "2.0", "id": 1,
        "method": "getTokenLargestAccounts",
        "params": [mint]
    }, timeout=30)
    raw = resp.json()
    if "error" in raw:
        return []
    accounts = raw.get("result", {}).get("value", [])[:limit]
    if not accounts:
        return []

    token_account_addresses = [a["address"] for a in accounts]
    resp2 = requests.post(rpc_url, json={
        "jsonrpc": "2.0", "id": 2,
        "method": "getMultipleAccounts",
        "params": [token_account_addresses, {"encoding": "jsonParsed"}]
    }, timeout=30)
    acct_infos = resp2.json().get("result", {}).get("value", [])

    owners = []
    for acct in acct_infos:
        if acct and acct.get("data", {}).get("parsed"):
            owner = acct["data"]["parsed"]["info"]["owner"]
        else:
            owner = "unknown"
        owners.append(owner)

    resp3 = requests.post(rpc_url, json={
        "jsonrpc": "2.0", "id": 3,
        "method": "getMultipleAccounts",
        "params": [owners, {"encoding": "base64"}]
    }, timeout=30)
    sol_infos = resp3.json().get("result", {}).get("value", [])

    holders = []
    for acct, owner, sol_acct in zip(accounts, owners, sol_infos):
        amount = float(acct.get("uiAmount") or 0)
        sol = (sol_acct["lamports"] / 1_000_000_000) if sol_acct else 0
        holders.append({"owner": owner, "tokens": amount, "sol": sol})
    return holders


def get_mints(chain_id: str = "solana", order_by: str = "liquidity") -> list[dict]:
    """
    Fetch pumpswap token mints from DexScreener, sorted by liquidity.

    Args:
        chain_id: "solana"
        order_by: "liquidity" (default) or "volume" - sort key

    Returns:
        List of dicts with mint, name, symbol, liquidity, market_cap, price
    """
    resp = requests.get(DEXSCREENER_URL, params={"q": DEX_ID}, timeout=30)
    resp.raise_for_status()
    data = resp.json()

    pairs = data.get("pairs", [])
    pairs = [p for p in pairs if p.get("dexId") == DEX_ID and p.get("chainId") == chain_id]

    if order_by == "liquidity":
        pairs.sort(key=lambda p: float(p.get("liquidity", {}).get("usd", 0) or 0), reverse=True)
    elif order_by == "volume":
        pairs.sort(key=lambda p: float(p.get("volume", {}).get("h24", 0) or 0), reverse=True)

    result = []
    for p in pairs:
        base = p.get("baseToken", {})
        if not base.get("address"):
            continue
        result.append({
            "mint": base["address"],
            "name": base.get("name", ""),
            "symbol": base.get("symbol", ""),
            "liquidity": float(p.get("liquidity", {}).get("usd", 0) or 0),
            "market_cap": float(p.get("marketCap", 0) or p.get("fdv", 0) or 0),
            "price": float(p.get("priceUsd", 0) or 0),
        })
    return result


if __name__ == "__main__":
    tokens = get_mints(order_by="liquidity")
    total_balance = {}
    print(f"Fetching holders for top {min(30, len(tokens))} tokens...\n")

    total_balance_sol = {}
    for t in tokens[:30]:
        holders = get_top_holders(t["mint"])
        t["holders"] = holders
        total_balance[t["mint"]] = sum(h["tokens"] for h in holders)
        total_balance_sol[t["mint"]] = sum(h["sol"] for h in holders)
        t["total_balance"] = total_balance[t["mint"]]
        t["total_balance_sol"] = total_balance_sol[t["mint"]]
        price = t["price"] or 1e-9
        t["sol_per_price"] = total_balance_sol[t["mint"]] / price

    tokens = sorted(tokens[:30], key=lambda t: t["total_balance_sol"], reverse=True)
    print(f"{'#':<4} {'Symbol':<12} {'Address':<45} {'Total %':>10} {'Total SOL':>10} {'Market Cap':>14}")
    print("-" * 100)
    for i, t in enumerate(tokens, 1):
        tb = total_balance.get(t["mint"], 0)
        total_pct = tb / 1e9
        tbsol = total_balance_sol.get(t["mint"], 0)
        mcap = t.get("market_cap", 0) or t.get("fdv", 0)
        print(f"{i:<4} {t['symbol']:<12} {t['mint']:<45} {total_pct:>9.2%}  {tbsol:>8.4f}  ${mcap:,.0f}")
