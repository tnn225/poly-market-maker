# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Automated market maker keeper for Polymarket CLOB (Central Limit Order Book). Places and cancels orders to maintain liquidity on prediction markets using configurable strategies.

**Status**: Experimental, active development. Python 3.10 required.

## Commands

```bash
# Setup
./install.sh              # Create venv and install dependencies
./install-dev.sh          # Install with dev tools (pytest, black, flake8)

# Running
./run-local.sh            # Run locally
docker compose up         # Run with Docker
python -m poly_market_maker [args]  # Direct execution

# Development
make test                 # Run pytest
make fmt                  # Format with black
pytest                    # Run tests
flake8                    # Lint
```

## Architecture

```
App (app.py) - Entry point, orchestrates components
    ├── Lifecycle (lifecycle.py) - Signal handling, sync loop
    ├── StrategyManager (strategy.py) - Strategy coordination
    │   ├── PriceEngine (price_engine.py) - WebSocket for BTC/USD prices
    │   ├── OrderBookEngine (order_book_engine.py) - WebSocket for order book
    │   ├── Strategy (strategies/) - AMM or Bands trading logic
    │   └── OrderBookManager (orderbook.py) - Order lifecycle
    └── ClobApi (clob_api.py) - Polymarket CLOB API wrapper
        └── Contracts (contracts.py) - ERC20/ERC1155 blockchain ops
```

**Main loop** (every `sync_interval` seconds):
1. Fetch current price from WebSocket/CLOB
2. Strategy computes expected orders
3. Compare with open orders
4. Cancel outdated, place new orders via blockchain

## Key Files

- `poly_market_maker/__main__.py` - Entry point
- `poly_market_maker/app.py` - Main App class
- `poly_market_maker/strategy.py` - StrategyManager
- `poly_market_maker/strategies/amm_strategy.py` - AMM strategy (concentrated liquidity)
- `poly_market_maker/strategies/bands_strategy.py` - Bands strategy (margin-based)
- `config/amm.json`, `config/bands.json` - Strategy configurations

## Configuration

**`.env`** - Credentials (see `.env.example`):
- `PRIVATE_KEY` - Wallet key for signing
- `RPC_URL` - Polygon RPC endpoint
- `CLOB_API_URL` - Polymarket CLOB API

**`config.env`** - Strategy selection:
- `STRATEGY` - "amm" or "bands"
- `CONFIG` - Path to strategy config JSON

## Strategies

**AMM**: Emulates concentrated liquidity pool. Places orders in bands around midpoint price. Params: `spread`, `delta`, `depth`, `max_collateral`.

**Bands**: Places orders at configurable margin distances from midpoint. Multiple bands per side with min/max/avg amounts.

See `docs/strategies/` for detailed documentation.

## External Dependencies

- Polymarket CLOB API: `https://clob.polymarket.com`
- WebSocket: `wss://ws-subscriptions-clob.polymarket.com`
- Polygon network (chain ID 137)
- Key libraries: `web3.py`, `py_clob_client`, `scikit-learn`
