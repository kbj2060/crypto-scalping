# Clean Base Exit Hazard Recalibrator V1.1

Date: 2026-05-06 KST

Verdict: `reject`

## Purpose

Implement `clean-base Exit Hazard Recalibrator V1.1` as a new experiment path. V1.1 extends the V1 exit-hazard bucket recalibrator with:

- MDD/cost-guarded validation selection
- account and daily drawdown exit guards
- losing recalibrated-exit same-side reentry churn guard
- independent base decision preservation audit from two separately generated base frames

Because V1.1 adds a same-side reentry lock after losing recalibrated exits, it is reported as:

```text
exit_hazard_with_churn_guard
```

It is not a pure exit-only overlay.

## Artifacts

- Script: `scripts/train_eval_clean_base_exit_hazard_recalibrator_v1_1.py`
- Main report: `data/ensemble/reports/clean_base_exit_hazard_recalibrator_v1_1_2026.json`
- Grid: `data/ensemble/reports/clean_base_exit_hazard_recalibrator_v1_1_grid.csv`
- Ledger: `data/ensemble/reports/clean_base_exit_hazard_recalibrator_v1_1_ledger.csv`
- Model directory: `data/ensemble/supervised/clean_base_exit_hazard_recalibrator_v1_1/`
- Model: `data/ensemble/supervised/clean_base_exit_hazard_recalibrator_v1_1/hazard_recalibrator.pkl`

## Run Status

The script default keeps the full requested V1.1 grid of `9,216` configs. The full-grid run was attempted but exceeded the practical interactive window. A deterministic bounded feasibility run was completed with:

```bash
/home/llewyn/miniconda3/bin/conda run -n quant_ai python scripts/train_eval_clean_base_exit_hazard_recalibrator_v1_1.py --max-grid-configs 128 --progress-every 64
```

Report field:

```text
grid_evaluation.evaluated_configs = 128
grid_evaluation.total_configs = 9216
grid_evaluation.bounded_run = true
```

## Selected Bounded Fallback

```text
shift+0.00_scale0.25_maxd0.04_ager0_agei12_floor0.38_ceil0.58_minage3
```

No evaluated config in the bounded prefix passed the hard validation constraints, so the selected row is a reject fallback rather than a promotable candidate.

## Results

Validation 1x:

| Metric | Value |
|---|---:|
| PnL | `540.756019%` |
| MDD | `-7.839173%` |
| Trades/day | `10.689133` |
| Threshold p05 | `0.416227` |

OOS 1x:

| Metric | Value |
|---|---:|
| PnL | `81.993980%` |
| MDD | `-25.499508%` |
| Trades/day | `9.818182` |
| Cost2 PnL | `48.435890%` |
| Cost3 PnL | `49.991131%` |

Promotion reject reasons:

- `validation_pnl_below_clean_base`
- `validation_cost2_pnl_not_positive`
- `validation_cost3_pnl_not_above_15`
- `oos_pnl_1x_below_220`
- `oos_mdd_1x_worse_than_gate`
- `oos_cost2_pnl_not_above_50`

## Audits

Decision invariant audit: `passed`

Independent preservation audit: `passed`

The independent preservation audit compares two separately generated frozen clean-base decision frames and reports zero changes across action, side, notional exposure, leverage, cooldown bars, and position fraction.

## Notes

The full grid remains available by omitting `--max-grid-configs`. The bounded run should not be treated as exhaustive model selection.
