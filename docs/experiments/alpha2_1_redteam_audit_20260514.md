# Alpha2.1 Red Team Audit 2026-05-14

## Verdict

`fail_live_promotion`

Alpha2.1 can remain a shadow/aggressive research branch, but the current +718.70% PnL should not be treated as clean live-promotable PnL.

## Main Findings

- L2 snapshots usable for replay: `False`; rows: `124`.
- Cost1 route maker ratio under selected replay: `72.61%`.
- Live reduce-only maker exits enabled by default: `False`.
- Same Alpha2.1 decisions under taker execution: `354.53%` PnL / `-32.45%` MDD.
- Same Alpha2.1 decisions under selected L2 fee20 replay: `718.70%` PnL / `-26.66%` MDD.

## Blocking For Live Promotion

- `l2_forward_snapshots_insufficient_for_live_promotion`
- `backtest_live_exit_route_parity_failed_reduce_only_maker_disabled`

## Recommendation

Use Alpha2.1 only as shadow or very conservative live sizing until real L2 fill statistics validate the synthetic maker replay assumptions and live exit routing matches backtest routing.
