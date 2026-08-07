# Trading Bot Alpha1.4 Execution Router Audit

Date: 2026-05-13 KST

## Scope

`trading_bot.py` keeps the existing Alpha1/V31 live decision stack and adds the Alpha1.4 execution layer to the Binance execution adapter.

The execution router does not change model decisions:

- Parent: `hf_v13_clean_regime_margin110_20260511`
- Add-on: `V21.2 Jackpot Runner`
- CASH scout: frozen `V27 Deep Scout`
- Deep scout exit: `V31 Rule Exit Overlay`
- Deep scout notional override: `2.0`

The new layer only changes how `BinanceFuturesExecutionAdapter` submits executable orders.

## Router Contract

Default runtime:

```text
BINANCE_EXECUTION_ALPHA14_ROUTER_ENABLE = true
BINANCE_EXECUTION_MAKER_REDUCE_ONLY_ENABLE = false
BINANCE_EXECUTION_MAKER_FALLBACK_MARKET = true
BINANCE_EXECUTION_MAKER_WAIT_SEC = 2.0
BINANCE_EXECUTION_MAKER_BOOK_DEPTH = 20
BINANCE_EXECUTION_MAKER_MAX_SPREAD_BPS = 4.0
BINANCE_EXECUTION_MAKER_MIN_IMBALANCE = 0.05
BINANCE_EXECUTION_MAKER_MIN_MICROPRICE_EDGE_BPS = 0.0
```

Routing logic:

1. Fetch or reuse decision-time order book snapshot.
2. If spread is too wide, use market order.
3. If same-side depth imbalance or microprice edge is favorable, submit post-only limit order:
   - buy: best bid
   - sell: best ask
   - `timeInForce=GTX`
   - `postOnly=true`
4. If live post-only order is not filled within `BINANCE_EXECUTION_MAKER_WAIT_SEC`, cancel it.
5. If fallback is enabled, submit remaining amount as market order.

Reduce-only orders stay market by default so exits are not delayed.

## Safety Gates

Existing execution gates remain unchanged:

- `BINANCE_EXECUTION_ENABLED=false` by default
- `BINANCE_EXECUTION_DRY_RUN=true` by default
- mainnet live orders still require `BINANCE_EXECUTION_CONFIRM_LIVE=I_UNDERSTAND_REAL_ORDERS`
- testnet requirement still defaults to enabled

## Validation

Passed:

- `python -m py_compile trading_bot_modules/binance_execution.py trading_bot.py`
- dry-run favorable book routes to post-only limit
- dry-run unfavorable/wide-spread book routes to market fallback

## Remaining Risks

- This implements live routing mechanics, not a full historical L2 queue-position simulator.
- Actual maker fills still depend on exchange queue priority and latency.
- The order book decision snapshots in `microstructure.duckdb.orderbook_decision_snapshots` should be collected during shadow runs before enabling real mainnet execution.
