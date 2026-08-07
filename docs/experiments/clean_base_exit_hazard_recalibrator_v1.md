# Clean Base Exit Hazard Recalibrator V1

Date: 2026-05-06 KST

Verdict: `implemented_but_reject_for_mdd`

## Purpose

Implement candidate 2:

```text
clean-base Exit-Only Hazard Recalibrator v1
```

The experiment preserves the frozen clean base policy and frozen clean base exit governor. It only recalibrates exit threshold and effective minimum exit age using train-only state buckets.

Hard constraints:

- no entry action changes
- no side changes
- no notional changes
- no leverage changes
- no cooldown changes
- no 2026 threshold selection

## Artifacts

- Script: `scripts/train_eval_clean_base_exit_hazard_recalibrator_v1.py`
- Main report: `data/ensemble/reports/clean_base_exit_hazard_recalibrator_v1_2026.json`
- Grid: `data/ensemble/reports/clean_base_exit_hazard_recalibrator_v1_grid.csv`
- Model directory: `data/ensemble/supervised/clean_base_exit_hazard_recalibrator_v1/`
- Model: `data/ensemble/supervised/clean_base_exit_hazard_recalibrator_v1/hazard_recalibrator.pkl`

## Data Split

| Split | Range | Rows | Use |
|---|---|---:|---|
| Train | `2025-01-01 00:00:00` to `2025-10-31 23:55:00` | `87496` | Fit exit-hazard bucket recalibrator only |
| Validation | `2025-11-01 00:00:00` to `2025-12-31 23:55:00` | `17568` | Select recalibration runtime config |
| OOS | `2026-01-01 00:00:00` to `2026-02-28 16:00:00` | `16897` | One-shot evaluation |

## Method

The trainer samples active clean-base lifecycle states from the training split and reuses the frozen clean exit-governor sample labeling objective. It buckets state by:

- side
- age bucket
- unrealized PnL bucket
- drawdown-from-peak bucket
- funding/liquidity/tail stress bucket
- current base signal alignment

Each bucket stores a smoothed train-only exit-label rate. At runtime, the recalibrator maps the active position state to a bucket, shifts the frozen exit-governor threshold around the clean-base threshold, and optionally adjusts min exit age. The selected V1 config used only threshold recalibration:

```text
shift+0.00_scale1.00_maxd0.14_ager0_agei0
```

## Results

Clean base OOS reference:

| Metric | Value |
|---|---:|
| PnL | `177.329809%` |
| MDD | `-17.759665%` |
| Trades | `363` |
| Trades/day | `6.187500` |

Selected recalibrator validation:

| Metric | Value |
|---|---:|
| PnL | `815.197282%` |
| MDD | `-11.094754%` |
| Trades | `709` |
| Trades/day | `11.623612` |

Selected one-shot 2026 OOS:

| Metric | Value |
|---|---:|
| PnL | `287.412074%` |
| MDD | `-21.046279%` |
| Trades | `613` |
| Trades/day | `10.448864` |
| Avg notional | `0.661041` |
| Avg leverage | `1.575337` |

Cost stress:

| Cost | PnL | MDD | Trades/day |
|---|---:|---:|---:|
| 1x | `287.412074%` | `-21.046279%` | `10.448864` |
| 2x | `55.969137%` | `-23.823092%` | `9.698864` |
| 3x | `0.418633%` | `-20.349429%` | `8.386364` |

Invariant audit:

```text
passed = true
```

## Interpretation

The recalibrator lifted OOS PnL and trade frequency, but it failed the clean-base promotion gate because MDD worsened from `-17.759665%` to `-21.046279%`. The 3x cost result is barely positive and does not provide enough robustness margin.

Decision: do not promote V1.
