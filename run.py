import numpy as np
import pandas as pd

# AMM parameters
b = 10
q_no = 0
prices = np.round(np.arange(0.01, 1.0, 0.01), 2)
mid_price = 0.50

# Compute q_yes for each price
q_yes = b * np.log(prices / (1 - prices)) + q_no

# Compute size between consecutive prices
sizes = np.diff(q_yes, prepend=q_yes[0])  # size at each price level

# Separate bid (prices < mid_price) and ask (prices > mid_price)
bid_mask = prices < mid_price
ask_mask = prices > mid_price

# Build bid table
bid_prices = prices[bid_mask][::-1]  # descending for bids
bid_sizes = -sizes[bid_mask][::-1]   # negative because AMM buys YES
bid_cum = np.cumsum(bid_sizes)
bid_table = pd.DataFrame({
    "price": bid_prices,
    "size": np.round(bid_sizes, 2),
    "cumulative": np.round(bid_cum, 2)
})

# Build ask table
ask_prices = prices[ask_mask]        # ascending for asks
ask_sizes = sizes[ask_mask]          # positive because AMM sells YES
ask_cum = np.cumsum(ask_sizes)
ask_table = pd.DataFrame({
    "price": ask_prices,
    "size": np.round(ask_sizes, 2),
    "cumulative": np.round(ask_cum, 2)
})

# Display
print("=== BID SIDE (AMM Buys YES) ===")
print(bid_table)
print("\n=== ASK SIDE (AMM Sells YES) ===")
print(ask_table)
