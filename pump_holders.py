import os
import requests
from dotenv import load_dotenv
load_dotenv()

HELIUS_KEY = os.getenv("HELIUS_KEY")
RPC = f"https://rpc.helius.xyz/?api-key={HELIUS_KEY}"
mint = "a3W4qutoEJA4232T2gwZUfgYJTetr96pU4SJMwppump"

# getTokenLargestAccounts returns top 20 token accounts (public RPC supports this)
resp = requests.post(RPC, json={
    "jsonrpc": "2.0", "id": 1,
    "method": "getTokenLargestAccounts",
    "params": [mint]
}, timeout=30)

raw = resp.json()
if "error" in raw:
    print(f"RPC error: {raw['error']}")
accounts = raw.get("result", {}).get("value", [])
print(f"Top holders fetched: {len(accounts)}")

token_account_addresses = [a["address"] for a in accounts]

# Resolve owner of each token account
resp2 = requests.post(RPC, json={
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

# Batch fetch SOL balances
resp3 = requests.post(RPC, json={
    "jsonrpc": "2.0", "id": 3,
    "method": "getMultipleAccounts",
    "params": [owners, {"encoding": "base64"}]
}, timeout=30)

sol_infos = resp3.json().get("result", {}).get("value", [])

print(f"\n{'#':<4} {'Owner':<45} {'Tokens':>15} {'SOL':>10}")
print("-" * 78)
for i, (acct, owner, sol_acct) in enumerate(zip(accounts, owners, sol_infos), 1):
    amount = float(acct.get("uiAmount") or 0)
    sol = (sol_acct["lamports"] / 1_000_000_000) if sol_acct else 0
    print(f"{i:<4} {owner:<45} {amount:>15,.2f}  {sol:>8.4f} SOL")
