# Clean Base Hybrid Lifecycle Exit V2

Date: 2026-05-06 KST

Verdict: `reject`

## Purpose

Implement a new experiment path for:

```text
clean-base hybrid lifecycle + exit-hazard editor v2
```

The experiment keeps the clean base policy and clean base exit governor frozen. Exit hazard recalibrator V1 is used only as feature generation: bucket, support, hazard rate, threshold delta versus global hazard, and exit governor probability. It does not directly replace or recalibrate the frozen exit governor.

The V2 action space is:

```text
NOOP
EARLY_EXIT
HOLD_LOCK_12
REDUCE_25
REDUCE_50
```

Forbidden mutations remain blocked by construction and audit:

- no new entry
- no entry deletion
- no side flip
- no notional increase
- no leverage increase
- no cooldown mutation
- `effective_notional <= base_notional`

## Artifacts

- Script: `scripts/train_eval_clean_base_hybrid_lifecycle_exit_v2.py`
- Main report: `data/ensemble/reports/clean_base_hybrid_lifecycle_exit_v2_2026.json`
- Grid: `data/ensemble/reports/clean_base_hybrid_lifecycle_exit_v2_grid.csv`
- Model directory: `data/ensemble/supervised/clean_base_hybrid_lifecycle_exit_v2/`
- Model: `data/ensemble/supervised/clean_base_hybrid_lifecycle_exit_v2/hybrid_lifecycle_exit_v2.pkl`
- Ledger: not produced

## Data Split

| Split | Range | Rows | Use |
|---|---|---:|---|
| Train | `2025-01-01 00:00:00` to `2025-10-31 23:55:00` | `87496` | Train hazard feature buckets only |
| Validation | `2025-11-01 00:00:00` to `2025-12-31 23:55:00` | `17568` | Select V2 lifecycle config |
| OOS | `2026-01-01 00:00:00` to `2026-02-28 16:00:00` | `16897` | One-shot OOS |

## Method

V2 reconstructs the clean-base admitted trade plan, then applies fixed-trade lifecycle actions only inside those trades. The selected validation config was:

```text
eep0.50_eh0.04_r25999.00_r50999.00_hold999.00_age12
```

This selected a conservative path: no entry-time reductions, no hold lock, and early exits only when train-only hazard features and the frozen exit governor probability both crossed the selected validation thresholds.

Selection score:

```text
pnl_1x + 0.35 * cost3_pnl - 10 * max(0, abs(mdd) - 17.76) - 15 * max(0, 5.8 - tpd)
```

## Results

Clean base OOS reference:

| Metric | Value |
|---|---:|
| PnL | `177.329809%` |
| MDD | `-17.759665%` |
| Trades/day | `6.187500` |

Selected validation:

| Cost | PnL | MDD | Trades/day |
|---|---:|---:|---:|
| 1x | `553.610081%` | `-12.694405%` | `11.394091` |
| 2x | `269.642709%` | `-12.973293%` | `11.394091` |
| 3x | `108.980653%` | `-17.225724%` | `11.394091` |

One-shot 2026 OOS:

| Cost | PnL | MDD | Trades/day |
|---|---:|---:|---:|
| 1x | `177.329809%` | `-17.759665%` | `6.187500` |
| 2x | `113.769852%` | `-18.314789%` | `6.187500` |
| 3x | `64.752636%` | `-20.096907%` | `6.187500` |

OOS action distribution:

| Action | Count |
|---|---:|
| NOOP | `320` |
| EARLY_EXIT | `43` |
| HOLD_LOCK_12 | `0` |
| REDUCE_25 | `0` |
| REDUCE_50 | `0` |

## Audit

Independent preservation audit:

```text
passed = true
entry_idx_changed = 0
side_changed = 0
cooldown_changed = 0
entry_deleted = 0
side_flip = 0
effective_exit_after_base_exit = 0
effective_notional_above_base = 0
leverage_changed = 0
```

Decision invariant audit: `passed = true`

Effective notional cap audit: `passed = true`

## Promotion Gate

Result: `failed`

Reject reasons:

```text
oos_pnl_below_220
cost2_pnl_below_120
```

Passed gate components:

- MDD is equal to the clean-base floor.
- Trades/day is above `5.8`.
- Cost3 PnL is above `60`.
- NOOP remains the largest action bucket.
- EARLY_EXIT is below `35%`.
- REDUCE_50 is below `10%`.
- Independent preservation and effective-notional cap audits pass.

Realistic replay/ledger was not run because canonical promotion gates did not pass.

