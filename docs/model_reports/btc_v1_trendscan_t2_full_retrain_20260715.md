# BTC v1 trend-scan t2 full retrain — 2026-07-15

## Outcome

The BTC v1 research candidate was fully retrained with the selected hourly
trend-scanning direction label (`abs(t) >= 2.0`). The candidate is rejected:
both the parent and the separately retrained risk sidecar lost money in the
fresh 2026 Q1 OOS period.

No live configuration or existing promoted artifact was changed.

## Training contract

- Direction target: BTC 1h trend-scanning, horizons 3/6/12/24/36/48 hours,
  selected by maximum absolute trend t-statistic, CASH when `abs(t) < 2.0`.
- 1h-to-5m mapping: `target_timestamp = ceil(feature_timestamp, 1h)`.
  The target timestamp is never before the 5m feature timestamp; delay is
  0–55 minutes.
- Quality target: existing BTC h48 conservative label (unchanged).
- Architecture: existing BTC v1 regime3 hard router plus three 3-head TabM
  experts (direction/quality/exit).
- Fit rows: complete pre-validation train split (78,624 rows); the prior
  30,000-row cap was disabled.
- Exit samples: complete generated exit path; the prior 12,000-row cap was
  disabled.
- Epochs: 4, seed 260620.
- Parent quality threshold: q0.55, selected from validation only.
- Risk sidecar: HGB, parent-output features, side-split, dynamic leverage,
  validation-only log-risk selection.

The inherited BTC v1 parent split starts validation at 2025-10-01, so its
validation period is 2025-10-01 through 2025-12-31 rather than the project
default 2025-09-01 through 2025-12-31. This divergence is explicit and is one
reason this run is research-only.

## Results

| Evaluation | PnL | MDD | Trades | Status |
|---|---:|---:|---:|---|
| Parent q0.55 validation | +6.98% | -1.71% | 44 | selected on validation |
| Parent q0.55 extended OOS (2026-01-01–2026-07-12) | -4.32% | -6.34% | 153 | fail |
| Parent + ATR/exit Q1 fresh-forward baseline | -5.37% | -6.55% | 48 | fail |
| Risk sidecar validation full replay | +26.70% | -5.88% | 32 | validation only |
| Risk sidecar Q1 fresh-forward full replay | -15.01% | -17.02% | 47 | fail |
| Risk sidecar extended OOS full replay | -19.05% | -26.48% | 91 | fail |

The risk sidecar amplified OOS loss because the validation-selected mapping
raised average Q1 notional to about 0.90 while the Q1 win rate was only 17.0%.

## Fresh-forward evidence

- `fresh_forward_bar_by_bar=true`
- `trade_ledgers_used_as_input=false`
- `saved_parent_exit_timestamps_used=false`
- `future_rows_used_for_entry=false`
- Q1 period: 2026-01-01 00:00:00 through 2026-03-31 23:55:00, 25,920 bars.
- Parent prediction artifacts were loaded at the exact q055 threshold and
  matched the runtime frame timestamps one-to-one.
- Trade ledgers written by the evaluator are outputs only and are not valid
  promotion inputs.

## Artifacts

- Hourly label pack: `tmp/causal_regen_20260516/btc_best_mean_pnl_trendscan_labels_20260715`
- 5m target adapter: `scripts/build_btc_v1_trendscan_5m_labels_20260715.py`
- Adapted labels: `tmp/causal_regen_20260516/btc_v1_trendscan_t2_5m_labels_20260715`
- Full parent: `tmp/causal_regen_20260516/btc_omega4_3head_parent72_loose_entry_quality_20260708_trendscan_t2_fulltrain_fullexit_20260715`
- Full-period risk sidecar: `tmp/causal_regen_20260516/btc_omega4_2_trade_risk_sidecar_20260708_trendscan_t2_fulltrain_fullexit_q055_20260715`
- Q1 fresh-forward evaluation: `tmp/causal_regen_20260516/btc_omega4_2_trade_risk_sidecar_20260708_trendscan_t2_fulltrain_fullexit_q055_q1fresh_20260715`

## Decision

`promotion_pass=false`. Keep the current BTC v1 live checkpoint unchanged.
The trend-scan label's oracle/backtest mean-PnL advantage did not survive
supervised learning and unseen Q1 execution costs.
