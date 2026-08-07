# Runtime-Native Training And Backtest Policy

This project must treat runtime-native evaluation as the promotion gate for all live trading models.

## Rule

All future training, model search, backtests, and live-promotion decisions must be validated through the same runtime path used by `trading_bot.py`.

The canonical validation path is:

1. Build or train the candidate model using strictly causal training data.
2. Inject the candidate artifact into `trading_bot.FinalGovernorRuntime`.
3. Run a runtime-native backtest that calls `FinalGovernorRuntime.decide()` sequentially.
4. Match live contracts for feature window, feature injection, router state, next-open execution, fee/slippage, leverage, notional exposure, and same-position resize accounting.
5. Promote only if the runtime-native result passes the agreed PnL/MDD/trade-count criteria.

## Non-Promotion Criteria

Standalone backtest functions are reference diagnostics only. Their PnL is not sufficient for promotion.

A candidate must not be promoted when:

- It only wins in a standalone evaluator.
- It has not been replayed through `FinalGovernorRuntime.decide()`.
- It uses a different feature window from live.
- It uses different AI feature injection or fallback behavior from live.
- It bypasses `GovernorPositionRouter` state assumptions.
- It omits next-open execution, fees, slippage, leverage, notional exposure, or resize costs.

## Current Canonical Script

Use `scripts/backtest_alpha3_runtime_native_20260515.py` as the current template.

If a new runtime-native evaluator is needed, it must preserve the same contract and explicitly document any intentional difference from live.

## Alpha3 CSV Loop Parity Exception

Alpha3 now has a separate debugging baseline:

- `docs/model_contracts/alpha3_csv_native_backtest_parity_20260516.md`
- `docs/experiments/alpha3_csv_native_parity_redteam_20260516.md`

This mode is intentionally not a pure live-runtime replay. It uses the live runner plumbing and report/ledger comparison, but executes the canonical CSV Alpha3 position loop so that action timestamps, actions, routes, and PnL can be matched exactly against the corrected CSV ledger.

Use this exception only when the goal is to freeze the historical CSV baseline and test whether a single layer changes behavior. Do not use it as a live promotion substitute.

## Reason

Previous Alpha3/V31 experiments showed high standalone backtest returns but failed when replayed through the actual live runtime. The root issue was a mismatch between independent evaluator assumptions and the real `trading_bot.py` decision/accounting path.

From this point forward, runtime-native backtest results are the source of truth.
