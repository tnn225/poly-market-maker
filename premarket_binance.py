import time
from datetime import datetime, timezone, timedelta
from poly_market_maker.order import Order, Side
from poly_market_maker.my_token import MyToken
from poly_market_maker.clob_api import ClobApi
from poly_market_maker.strategies.simple_order import SimpleOrder
from poly_market_maker.binance import Binance

MIN_SIZE = 10

clob_api = ClobApi()

class TradeManager:
    def __init__(self, interval: int):
        self.clob_api = clob_api
        self.market = clob_api.get_market(interval)
        self.interval = interval
        self.amm_a = SimpleOrder(token=MyToken.A)
        self.amm_b = SimpleOrder(token=MyToken.B)

    def place_order(self, new_order: Order) -> Order:
        order_id = self.clob_api.place_order(
            price=new_order.price,
            size=new_order.size,
            side=new_order.side.value,
            token_id=self.market.token_id(new_order.token),
        )
        return Order(
            price=new_order.price,
            size=new_order.size,
            side=new_order.side,
            id=order_id,
            token=new_order.token,
        )

    def trade(self, seconds_left: int, probability: float):
        print(f"trade interval: {self.interval} seconds_left: {seconds_left} probability: {probability}")
        if probability > 0.5699:
            self.orders = self.amm_a.get_buy_orders()
            for order in self.orders:
                print(f"trade delta: {probability} order: {order}")
                self.place_order(order)

        if probability < 0.4381:
            self.orders = self.amm_b.get_buy_orders()
            for order in self.orders:
                print(f"trade delta: {probability} order: {order}")
                self.place_order(order)


def get_probability(binance: Binance, model, feature_cols: list[str], seconds_left: int, interval: int):
    df = binance.get_df(start_time=datetime.now(timezone.utc) - timedelta(hours=1), end_time=datetime.now(timezone.utc))
    df = binance.add_features(df)
    df['probability'] = model.predict_proba(df[feature_cols])[:, 1]

    # df['interval'] = df['open_time'] + 900

    print(df[['open_time', 'probability', 'is_up', 'label']])
    # Return probability at open_time == interval
    row = df[df['open_time'] == interval-900] # preciou 15-min ago
    probability = row['probability'].iloc[0] if not row.empty else 0.5
    print(f"{seconds_left} interval: {interval} probability: {probability}")
    return probability

def main():
    interval = 0
    trade_manager = None
    has_orders = False
    probability = 0.5

    binance = Binance(symbol="BTCUSDT", interval="15m")
    train_df, test_df = binance.train_test_split()
    model = binance.get_model()
    feature_cols = binance.feature_cols

    while True:
        time.sleep(1)

        """
        data = price_engine.get_data()
        if data is None:
            print("No price data")
            continue

        price = data.get('price')
        target = data.get('target')
        if price is None or target is None:
            continue
        delta = price - target
        """
        now = int(time.time())
        seconds_left = 900 - (now % 900)
        if seconds_left % 60 == 0:
            probability = get_probability(binance, model, feature_cols, seconds_left, now // 900 * 900)
            # clob_api.print_holders(now // 900 * 900)

        if now // 900 * 900 > interval and seconds_left > 840:  # 15-min intervals

            interval = now // 900 * 900
            probability = get_probability(binance, model, feature_cols, seconds_left, interval) 

            trade_manager = TradeManager(interval)
            trade_manager.trade(seconds_left, probability)
            has_orders = True

        if trade_manager is not None and seconds_left < 600 and has_orders:
            trade_manager.clob_api.cancel_all_orders()
            has_orders = False

if __name__ == "__main__":
    main()
