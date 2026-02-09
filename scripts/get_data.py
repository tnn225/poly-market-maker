import requests
import pandas as pd
from datetime import datetime, timedelta
import matplotlib.pyplot as plt


def fetch_ethusdt_data():
    """
    Fetch Binance ETHUSDT futures data:
    - Price (15m candles)
    - Volume (15m candles)
    - Open Interest
    - Funding Rate
    Duration: Last 24 hours
    """

    # Calculate timestamps
    look_backs = 50
    start_time = int((datetime.now() - timedelta(days=look_backs + 10)).timestamp() * 1000)
    end_time = int((datetime.now() - timedelta(days=look_backs)).timestamp() * 1000)
    interval = '15m'
    limit = 1000

    print("Fetching Binance ETHUSDT Futures Data...")
    print(f"Period: Last 24 hours (15-minute intervals)\n")

    # 1. Fetch 15-minute klines (price & volume)
    print("📊 Fetching price and volume data...")
    klines_url = f"https://fapi.binance.com/fapi/v1/klines"
    klines_params = {
        'symbol': 'ETHUSDT',
        'interval': interval,
        'startTime': start_time,
        'endTime': end_time,
        'limit': limit
    }

    klines_response = requests.get(klines_url, params=klines_params)
    klines_data = klines_response.json()

    open_interest_url = f"https://fapi.binance.com/futures/data/openInterestHist"
    open_interest_params = {
        'symbol': 'ETHUSDT',
        'period': interval,
        'startTime': start_time,
        'endTime': end_time,
        'limit': limit
    }
    open_interest_response = requests.get(open_interest_url, params=open_interest_params)
    open_interest_data = open_interest_response.json()

    # Process klines data
    df = pd.DataFrame(klines_data, columns=[
        'timestamp', 'open', 'high', 'low', 'close', 'volume',
        'close_time', 'quote_volume', 'trades', 'taker_buy_base',
        'taker_buy_quote', 'ignore'
    ])

    df_oi = pd.DataFrame(open_interest_data, columns=[
        'timestamp', 'sumOpenInterest', 'sumOpenInterestValue', 'CMCCirculatingSupply'
    ])

    df = pd.merge(df, df_oi, on='timestamp', how='left')

    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df['price'] = df['close'].astype(float)
    df['volume'] = df['volume'].astype(float)
    df['sumOpenInterest'] = df['sumOpenInterest'].astype(float)
    df = df[['timestamp', 'price', 'volume', 'sumOpenInterest']]

    # 2. Fetch current Open Interest
    print("📈 Fetching open interest...")
    oi_url = "https://fapi.binance.com/fapi/v1/openInterest"
    oi_params = {'symbol': 'ETHUSDT'}
    oi_response = requests.get(oi_url, params=oi_params)
    oi_data = oi_response.json()
    open_interest = float(oi_data['openInterest'])

    # 3. Fetch Funding Rate
    print("💰 Fetching funding rate...")
    funding_url = "https://fapi.binance.com/fapi/v1/premiumIndex"
    funding_params = {'symbol': 'ETHUSDT'}
    funding_response = requests.get(funding_url, params=funding_params)
    funding_data = funding_response.json()
    funding_rate = float(funding_data['lastFundingRate']) * 100  # Convert to percentage

    # 4. Fetch 24h stats
    print("📊 Fetching 24h statistics...\n")
    ticker_url = "https://fapi.binance.com/fapi/v1/ticker/24hr"
    ticker_params = {'symbol': 'ETHUSDT'}
    ticker_response = requests.get(ticker_url, params=ticker_params)
    ticker_data = ticker_response.json()

    current_price = float(ticker_data['lastPrice'])
    price_change_24h = float(ticker_data['priceChangePercent'])
    volume_24h = float(ticker_data['volume'])

    # Display summary
    print("=" * 60)
    print("BINANCE ETHUSDT FUTURES - CURRENT METRICS")
    print("=" * 60)
    print(f"Current Price:       ${current_price:,.2f}")
    print(f"24h Change:          {price_change_24h:+.2f}%")
    print(f"Open Interest:       {open_interest:,.2f} ETH")
    print(f"Funding Rate:        {funding_rate:.4f}%")
    print(f"24h Volume:          {volume_24h:,.2f} ETH")
    print("=" * 60)
    print(f"\nData Points: {len(df)} candles (15-minute intervals)")
    print(f"Time Range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    print("\n")

    # Display first few rows
    print("Sample Data (First 5 candles):")
    print(df.head().to_string(index=False))
    print("\n")

    # Create visualizations
    print("Creating charts...")
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 12))

    # Price chart
    ax1.plot(df['timestamp'], df['price'], color='#3B82F6', linewidth=2)
    ax1.set_title('ETHUSDT Price (15m intervals)', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Price (USDT)', fontsize=12)
    ax1.grid(True, alpha=0.3)
    ax1.tick_params(axis='x', rotation=45)

    # Volume chart
    ax2.plot(df['timestamp'], df['volume'], color='#10B981', linewidth=2)
    ax2.set_title('Trading Volume (15m intervals)', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Time', fontsize=12)
    ax2.set_ylabel('Volume (ETH)', fontsize=12)
    ax2.grid(True, alpha=0.3)
    ax2.tick_params(axis='x', rotation=45)

    # OI chart
    ax3.plot(df['timestamp'], df['sumOpenInterest'], color='red', linewidth=2)
    ax3.set_title('Open Interest (15m intervals)', fontsize=14, fontweight='bold')
    ax3.set_xlabel('Time', fontsize=12)
    ax3.set_ylabel('Open Interest (ETH)', fontsize=12)
    ax3.grid(True, alpha=0.3)
    ax3.tick_params(axis='x', rotation=45)

    plt.tight_layout()
    plt.savefig('binance_bot/ethusdt_binance_futures.png', dpi=300, bbox_inches='tight')
    print("✅ Chart saved as 'ethusdt_binance_futures.png'")

    # Save data to CSV
    df.to_csv('binance_bot/ethusdt_data_15m.csv', index=False)
    print("✅ Data saved as 'ethusdt_data_15m.csv'")

    # Create summary file
    with open('binance_bot/ethusdt_summary.txt', 'w') as f:
        f.write("BINANCE ETHUSDT FUTURES SUMMARY\n")
        f.write("=" * 60 + "\n")
        f.write(f"Generated: {datetime.now()}\n\n")
        f.write(f"Current Price:       ${current_price:,.2f}\n")
        f.write(f"24h Change:          {price_change_24h:+.2f}%\n")
        f.write(f"Open Interest:       {open_interest:,.2f} ETH\n")
        f.write(f"Funding Rate:        {funding_rate:.4f}%\n")
        f.write(f"24h Volume:          {volume_24h:,.2f} ETH\n")
        f.write(f"\nLowest Price (24h):  ${df['price'].min():,.2f}\n")
        f.write(f"Highest Price (24h): ${df['price'].max():,.2f}\n")
        f.write(f"Average Volume:      {df['volume'].mean():,.2f} ETH\n")

    print("✅ Summary saved as 'ethusdt_summary.txt'")

    return df, {
        'price': current_price,
        'price_change_24h': price_change_24h,
        'open_interest': open_interest,
        'funding_rate': funding_rate,
        'volume_24h': volume_24h
    }


def fetch_btcusdt_data(days=30):
    """
    Fetch BTC/USDT 15m kline data from Binance Futures and save to CSV.

    Args:
        days: Number of days to fetch (default 30)

    Returns:
        DataFrame with the kline data
    """
    print(f"Fetching Binance BTCUSDT Futures 15m Kline Data for {days} days...")

    url = "https://api.binance.com/api/v1/klines"
    limit_per_request = 1500

    end_time = int(datetime.now().timestamp() * 1000)
    start_time = int((datetime.now() - timedelta(days=days)).timestamp() * 1000)

    all_data = []
    current_start = start_time

    while current_start < end_time:
        params = {
            "symbol": "BTCUSDT",
            "interval": "15m",
            "startTime": current_start,
            "endTime": end_time,
            "limit": limit_per_request
        }

        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()

        if not data:
            break

        all_data.extend(data)
        # Move start to after last candle
        current_start = data[-1][6] + 1  # close_time + 1ms
        print(f"  Fetched {len(data)} candles, total: {len(all_data)}")

    # Binance kline format:
    # [0] Open time, [1] Open, [2] High, [3] Low, [4] Close, [5] Volume,
    # [6] Close time, [7] Quote asset volume, [8] Number of trades,
    # [9] Taker buy base asset volume, [10] Taker buy quote asset volume, [11] Ignore

    records = []
    for kline in all_data:
        timestamp = datetime.fromtimestamp(kline[0] / 1000).strftime('%Y-%m-%d %H:%M:%S')
        open_price = float(kline[1])
        close_price = float(kline[4])
        volume_btc = float(kline[5])
        volume_usd = float(kline[7])  # Quote asset volume (USDT)

        records.append({
            "timestamp": timestamp,
            "open_price": open_price,
            "close_price": close_price,
            "volume_btc": volume_btc,
            "volume_usd": volume_usd
        })

    df = pd.DataFrame(records)

    # Save to CSV
    output_file = "./data/btcusdt_data_15m.csv"
    df.to_csv(output_file, index=False)
    print(f"Saved {len(df)} records to {output_file}")
    print(f"Time Range: {df['timestamp'].iloc[0]} to {df['timestamp'].iloc[-1]}")

    return df


if __name__ == "__main__":
    df = fetch_btcusdt_data(days=365)
    # df, metrics = fetch_ethusdt_data()
    print("\n✅ All data fetched successfully!")
    # exec(open("binance_bot/scalping_signals.py").read())
    # exec(open("binance_bot/backtest_strategy.py").read())
