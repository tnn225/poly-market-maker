import time

from poly_market_maker.intervals import Interval

def main():
    """Test function to fetch target prices."""
    intervals = Interval()

    while True:
        now = int(time.time()) - 900 # 15 minutes ago
        interval = now // 900 * 900

        try:
            for symbol in ["BTC", "ETH", "SOL", "XRP"]:
                data = intervals.get_data(symbol, interval)
                if data is not None:
                    print(f"{symbol} {interval}, openPrice: {data.get('openPrice')}, closePrice: {data.get('closePrice')}")
                else:
                    print(f"{symbol} {interval} - no data")
        except Exception as e:
            print(f"Error fetching data for interval {interval}: {e}")
            time.sleep(60)

        time.sleep(60 * 3)
        
if __name__ == "__main__":
    main()
