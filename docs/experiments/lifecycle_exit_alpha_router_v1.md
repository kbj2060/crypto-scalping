# lifecycle_exit_alpha_router_v1

## Summary

Implemented as a per-trade router approximation over the fixed clean-base trade plan. Lifecycle V1 is the default path. Exit V1 alpha timing is only considered when Lifecycle V1 did not already exit earlier than the base trade.

## Selected Config

- p0.55_cap0.10_acct0.06_day0.012_lock24
- Validation rows: 432
- Selection score: 960.018411

## OOS Metrics

- PnL 1x: 207.236888
- MDD 1x: -18.016318
- Trades/day: 6.187500
- Cost2 PnL: 127.776479
- Cost3 PnL: 68.834031
- Alpha fraction: 0.000000
- Alpha MDD contribution: 0.000000

## Approximation Contract

The router uses entry-time per-trade features. Requested intra-trade fields such as age, unrealized PnL, peak unrealized PnL, and drawdown from trade peak are deterministic entry-time defaults in this v1. Runtime account_dd, daily_dd, loss_streak, and prior trade giveback are pre-decision replay state.

## Gates

- Verdict: reject_for_promotion_gate
- Promotion passed: False
- Shadow continue passed: False
- Reject reasons: OOS PnL >= 220, OOS MDD >= -17.759665

## Artifacts

- Report: `/home/llewyn/crypto-scalping/data/ensemble/reports/lifecycle_exit_alpha_router_v1_2026.json`
- Grid: `/home/llewyn/crypto-scalping/data/ensemble/reports/lifecycle_exit_alpha_router_v1_grid.csv`
- Ledger: `/home/llewyn/crypto-scalping/data/ensemble/reports/lifecycle_exit_alpha_router_v1_ledger.csv`
- Model: `/home/llewyn/crypto-scalping/data/ensemble/supervised/lifecycle_exit_alpha_router_v1/alpha_router.pkl`
