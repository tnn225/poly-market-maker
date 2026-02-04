"""
Read holders CSV; for each interval (at highest timestamp), count shares
Up/Down that are same vs different from two reference wallets.
"""
import pandas as pd
from pathlib import Path

# CSV path: under repo root; file lives in data/holders/
CSV_PATH = Path(__file__).resolve().parents[1] / "data" / "holders" / "holders_btc_2026-01-30.csv"

WALLET_A = "0xdb5784453ffa8a03a1024c031f0c60ec96fdb0e0"
WALLET_B = "0xdb5784453ffa8a03a1024c031f0c60ec96fdb0e0"


def main() -> None:
    df = pd.read_csv(CSV_PATH)
    df["proxyWallet"] = df["proxyWallet"].str.strip().str.lower()
    wallet_a = WALLET_A.lower()
    wallet_b = WALLET_B.lower()

    # For each interval, keep only rows with the highest timestamp
    max_ts = df.groupby("interval")["timestamp"].transform("max")
    latest = df[df["timestamp"] == max_ts].copy()

    # For each interval, get the set of sides (Up/Down) held by the two wallets
    def wallet_sides(g: pd.DataFrame) -> set:
        w = g[g["proxyWallet"].isin({wallet_a, wallet_b})]
        return set(w["side"].unique())

    # For each interval: sum shares where side matches wallets (same) vs not (different).
    # Skip intervals where neither wallet has a side.
    # Also collect (wallet, amount) for "same" and "different" sides per wallet.
    rows = []
    same_by_wallet: list[pd.DataFrame] = []
    different_by_wallet: list[pd.DataFrame] = []
    for interval, grp in latest.groupby("interval"):
        sides = wallet_sides(grp)
        if not sides:
            continue
        same_mask = grp["side"].isin(sides)
        same_grp = grp.loc[same_mask, ["proxyWallet", "amount"]].copy()
        same_grp["interval"] = interval
        same_by_wallet.append(same_grp)
        diff_grp = grp.loc[~same_mask, ["proxyWallet", "amount"]].copy()
        diff_grp["interval"] = interval
        different_by_wallet.append(diff_grp)
        same_shares = same_grp["amount"].sum()
        different_shares = diff_grp["amount"].sum()
        rows.append({
            "interval": interval,
            "timestamp": grp["timestamp"].iloc[0],
            "wallet_sides": "|".join(sorted(sides)),
            "shares_same": round(same_shares),
            "shares_different": round(different_shares),
        })

    out = pd.DataFrame(rows)
    print(out.to_csv(index=False))

    # Per-wallet totals: shares_same and different_shares
    if same_by_wallet or different_by_wallet:
        same_df = (
            pd.concat(same_by_wallet, ignore_index=True)
            .groupby("proxyWallet", as_index=False)["amount"]
            .sum()
            .rename(columns={"amount": "shares_same"})
        ) if same_by_wallet else pd.DataFrame(columns=["proxyWallet", "shares_same"])
        diff_df = (
            pd.concat(different_by_wallet, ignore_index=True)
            .groupby("proxyWallet", as_index=False)["amount"]
            .sum()
            .rename(columns={"amount": "different_shares"})
        ) if different_by_wallet else pd.DataFrame(columns=["proxyWallet", "different_shares"])
        top_wallets = diff_df.merge(same_df, on="proxyWallet", how="outer").fillna(0)
        top_wallets["different_shares"] = top_wallets["different_shares"].round().astype(int)
        top_wallets["shares_same"] = top_wallets["shares_same"].round().astype(int)
        top_wallets = top_wallets.sort_values("different_shares", ascending=False)
        print("Top wallets by different_shares (opposite side from ref):")
        print(top_wallets.head(20).to_csv(index=False))

        # Wallets on same side as ref (shares_same > 0), top by shares_same
        same_side = (
            top_wallets[top_wallets["shares_same"] > 0]
            .sort_values("shares_same", ascending=False)
            .head(20)
        )
        print("Top wallets with same side as ref (by shares_same):")
        print(same_side.to_csv(index=False))

    return out


if __name__ == "__main__":
    main()
