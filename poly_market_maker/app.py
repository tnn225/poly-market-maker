import logging
from prometheus_client import start_http_server
import os  
import time

from poly_market_maker.args import get_args
from poly_market_maker.order_book_engine import OrderBookEngine
from poly_market_maker.price_engine import PriceEngine
from poly_market_maker.price_feed import PriceFeedClob
from poly_market_maker.gas import GasStation, GasStrategy
from poly_market_maker.prediction_engine import PredictionEngine
from poly_market_maker.utils import setup_logging, setup_web3
from poly_market_maker.order import Order, Side
from poly_market_maker.market import Market
from poly_market_maker.my_token import MyToken, Collateral
from poly_market_maker.clob_api import ClobApi
from poly_market_maker.lifecycle import Lifecycle
from poly_market_maker.orderbook import OrderBookManager
from poly_market_maker.contracts import Contracts
from poly_market_maker.metrics import keeper_balance_amount
from poly_market_maker.strategy import StrategyManager

from dotenv import load_dotenv          # Environment variable management
load_dotenv()                           # Load environment variables from .env file

FUNDER = os.getenv("FUNDER")
TARGET = os.getenv("TARGET")

class App:
    """Market maker keeper on Polymarket CLOB"""

    def __init__(self, args: list):
        setup_logging()
        self.logger = logging.getLogger(__name__)

        args = get_args(args)
        self.sync_interval = args.sync_interval
        self.refresh_frequency = args.refresh_frequency

        # server to expose the metrics.
        self.metrics_server_port = args.metrics_server_port
        start_http_server(self.metrics_server_port)

        self.web3 = setup_web3(args.rpc_url, args.private_key)
        # self.address = self.web3.eth.account.from_key(args.private_key).address
        self.address = FUNDER

        self.clob_api = ClobApi()

        self.gas_station = GasStation(
            strat=GasStrategy(args.gas_strategy),
            w3=self.web3,
            url=args.gas_station_url,
            fixed=args.fixed_gas_price,
        )
        self.contracts = Contracts(self.web3, self.gas_station)

        self.price_engine = PriceEngine(symbol="btc/usd")
        self.price_engine.start()

        self.order_book_engine = None

        self.interval = 0

        self.args_strategy = args.strategy
        self.args_strategy_config = args.strategy_config


        self.setup()

    """
    main
    """

    def main(self):
        self.logger.debug(self.sync_interval)
        with Lifecycle() as lifecycle:
            lifecycle.on_startup(self.startup)
            lifecycle.every(self.sync_interval, self.synchronize)  # Sync every 5s
            lifecycle.on_shutdown(self.shutdown)

    """
    lifecycle
    """

    def startup(self):
        self.logger.info("Running startup callback...")
        self.approve()
        time.sleep(5)  # 5 second initial delay so that bg threads fetch the orderbook
        self.logger.info("Startup complete!")

    def setup(self):
        interval = int(time.time()) // 900 * 900 

        if self.interval < interval:
            self.interval = interval
            self.market = self.clob_api.get_market(self.interval)

            data = self.price_engine.get_data()
            if data['interval'] != interval:
                raise Exception(f"No price target for {self.interval} {data}")

            self.price_feed = PriceFeedClob(self.market, self.clob_api)

            self.order_book_manager = OrderBookManager(
                self.refresh_frequency, max_workers=1
            )
            self.order_book_manager.get_orders_with(self.get_orders)
            self.order_book_manager.get_balances_with(self.get_balances)
            self.order_book_manager.cancel_orders_with(
                lambda order: self.clob_api.cancel_order(order.id)
            )
            self.order_book_manager.place_orders_with(self.place_order)
            self.order_book_manager.cancel_all_orders_with(
                lambda _: self.clob_api.cancel_all_orders()
            )
            self.order_book_manager.start()

            if self.order_book_engine:
                self.order_book_engine.stop()
            self.order_book_engine = OrderBookEngine(self.market) 
            self.order_book_engine.start()

            self.prediction_engine = PredictionEngine()

            self.strategy_manager = StrategyManager(
                self.args_strategy,
                self.args_strategy_config,
                self.price_feed,
                self.order_book_manager,
                self.price_engine,
                self.order_book_engine,
                self.prediction_engine
            )

    def synchronize(self):
        """
        Synchronize the orderbook by cancelling orders out of bands and placing new orders if necessary
        """
        self.logger.debug("Synchronizing orderbook...")

        self.setup()

        self.strategy_manager.synchronize()
        self.logger.debug("Synchronized orderbook!")

    def shutdown(self):
        """
        Shut down the keeper
        """
        self.logger.info("Keeper shutting down...")
        self.order_book_manager.cancel_all_orders()
        self.logger.info("Keeper is shut down!")

    """
    handlers
    """

    def get_balances(self) -> dict:
        """
        Fetch the onchain balances of collateral and conditional tokens for the keeper
        """
        self.logger.debug(f"Getting balances for address: {self.address}")

        collateral_balance = self.contracts.token_balance_of(
            self.clob_api.get_collateral_address(), self.address
        )
        token_A_balance = self.contracts.token_balance_of(
            self.clob_api.get_conditional_address(),
            self.address,
            self.market.token_id(MyToken.A),
        )
        token_B_balance = self.contracts.token_balance_of(
            self.clob_api.get_conditional_address(),
            self.address,
            self.market.token_id(MyToken.B),
        )
        gas_balance = self.contracts.gas_balance(self.address)

        keeper_balance_amount.labels(
            accountaddress=self.address,
            assetaddress=self.clob_api.get_collateral_address(),
            tokenid="-1",
        ).set(collateral_balance)
        keeper_balance_amount.labels(
            accountaddress=self.address,
            assetaddress=self.clob_api.get_conditional_address(),
            tokenid=self.market.token_id(MyToken.A),
        ).set(token_A_balance)
        keeper_balance_amount.labels(
            accountaddress=self.address,
            assetaddress=self.clob_api.get_conditional_address(),
            tokenid=self.market.token_id(MyToken.B),
        ).set(token_B_balance)
        keeper_balance_amount.labels(
            accountaddress=self.address,
            assetaddress="0x0",
            tokenid="-1",
        ).set(gas_balance)

        return {
            Collateral: collateral_balance,
            MyToken.A: token_A_balance,
            MyToken.B: token_B_balance,
        }

    def get_orders(self) -> list[Order]:
        orders = self.clob_api.get_orders(self.market.condition_id)
        return [
            Order(
                size=order_dict["size"],
                price=order_dict["price"],
                side=Side(order_dict["side"]),
                token=self.market.token(order_dict["token_id"]),
                id=order_dict["id"],
            )
            for order_dict in orders
        ]

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

    def approve(self):
        """
        Approve the keeper on the collateral and conditional tokens
        """
        collateral = self.clob_api.get_collateral_address()
        conditional = self.clob_api.get_conditional_address()
        exchange = self.clob_api.get_exchange()

        self.contracts.max_approve_erc20(collateral, self.address, exchange)
        self.contracts.max_approve_erc1155(conditional, self.address, exchange)
