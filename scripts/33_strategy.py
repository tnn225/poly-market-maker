from dataclasses import dataclass
from math import sqrt, log, exp
from typing import List, Dict

@dataclass
class Order:
    side: str          # "buy" or "sell"
    price: float       # rounded to 2
    size: float        # token0 size, rounded to 2

def _logspace(a: float, b: float, n: int) -> List[float]:
    la, lb = log(a), log(b)
    return [exp(la + (lb - la) * i / (n - 1)) for i in range(n)]

def _round2(x: float) -> float:
    return round(x, 2)

def v3_to_orderbook_rounded(
    L: float,
    bid: float,
    ask: float,
    p_low: float = 0.01,
    p_high: float = 0.99,
    n_levels_each_side: int = 80,
) -> List[Order]:
    if not (p_low <= bid <= p_high and p_low <= ask <= p_high):
        raise ValueError("Require p_low < p_current < p_high")

    # aggregate by (side, rounded_price)
    agg: Dict[str, Dict[float, float]] = {"buy": {}, "sell": {}}

    # ----- ASKS: sell token0 as price moves up -----
    ask_prices = []
    for i in range(len(ask_prices) - 1):
        p_a = ask_prices[i]
        p_b = ask_prices[i + 1]
        sp_a, sp_b = sqrt(p_a), sqrt(p_b)

        delta0 = L * (1.0 / sp_a - 1.0 / sp_b)  # token0 you sell
        size0 = max(0.0, delta0)

        price_r = _round2(p_b)
        size_r = _round2(size0)

        if price_r <= 0 or size_r <= 0:
            continue

        agg["sell"][price_r] = _round2(agg["sell"].get(price_r, 0.0) + size_r)

    # ----- BIDS: buy token0 as price moves down -----
    bid_prices = list(reversed(_logspace(p_low, p_current, n_levels_each_side)))
    for i in range(len(bid_prices) - 1):
        p_a = bid_prices[i]       # higher
        p_b = bid_prices[i + 1]   # lower
        sp_a, sp_b = sqrt(p_a), sqrt(p_b)

        delta0 = L * (1.0 / sp_b - 1.0 / sp_a)  # token0 you buy
        size0 = max(0.0, delta0)

        price_r = _round2(p_b)
        size_r = _round2(size0)

        if price_r <= 0 or size_r <= 0:
            continue

        agg["buy"][price_r] = _round2(agg["buy"].get(price_r, 0.0) + size_r)

    # build book: bids desc, asks asc
    bids = [Order("buy", p, s) for p, s in sorted(agg["buy"].items(), key=lambda x: x[0], reverse=True)]
    asks = [Order("sell", p, s) for p, s in sorted(agg["sell"].items(), key=lambda x: x[0])]
    return bids + asks

# Example
if __name__ == "__main__":
    book = v3_to_orderbook_rounded(L=500.0, p_current=0.75, n_levels_each_side=120)
    for o in book:
        print(o)
