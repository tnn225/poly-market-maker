import csv
from datetime import datetime

from poly_market_maker.clob_api import ClobApi
from poly_market_maker.utils.common import get_sifu_addresses

def run():
    addresses = get_sifu_addresses()
    clob_api = ClobApi()
    balance_by_wallet = {}
    today = datetime.now().strftime("%Y-%m-%d")
    for address in addresses:
        if address in balance_by_wallet:
            continue
        balance = clob_api.get_balance(address)
        balance_by_wallet[address] = balance
        print(f"{address}: {balance} USDC.e")
    with open(f'./data/balance/balance_{today}.csv', 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['address', 'balance'])
        for address, balance in sorted(
            balance_by_wallet.items(), key=lambda x: x[1], reverse=True
        ):
            writer.writerow([address, balance])

def main():
    run()
 
if __name__ == "__main__":
    main()