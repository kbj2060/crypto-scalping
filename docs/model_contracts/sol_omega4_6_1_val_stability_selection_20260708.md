# SOL Omega4.6.1 Validation-Only Stability Selection - 2026-07-08

## Purpose

Improve the SOL baseline drawdown profile without selecting parameters on OOS.

OOS was not used for parameter selection. The selected parameters were chosen
from the SOL `zig075 q070` validation ledger only, then OOS was reported once
as a sealed holdout check.

## Selection Rule

- Component: `zig075`
- Quality tag: `q070`
- Selection data: validation split only
- Scale grid: `0.5,0.75,1.0,1.25,1.5,1.75,2.0`
- Duration candidates: no gate plus validation `ou_halflife` quantiles `0.05..0.80`
- Minimum validation trades after gate: `24`
- Validation MDD budget: `>= -20%`
- Monthly validation MDD budget: `>= -12%`
- Worst validation month PnL budget: `>= -8%`
- Objective after gates: validation stability score, not OOS PnL/MDD

## Selected Parameters

- `long_scale=0.5`
- `short_scale=1.75`
- `duration_threshold=0.0055208323`
- `duration_quantile=0.30`

Validation-only search artifact:

- `tmp/causal_regen_20260516/sol_omega4_6_1_val_stability_search_20260708/report.json`
- `tmp/causal_regen_20260516/sol_omega4_6_1_val_stability_search_20260708/candidate_grid.csv`

Exact replay artifact:

- `tmp/causal_regen_20260516/sol_val_stability_exact_20260708/report.json`
- `tmp/causal_regen_20260516/sol_val_stability_exact_20260708/validation_ledger.csv`
- `tmp/causal_regen_20260516/sol_val_stability_exact_20260708/oos_ledger.csv`

## Exact Replay Result

| model | split | PnL | MDD | trades | WR |
|---|---|---:|---:|---:|---:|
| previous baseline `1.0/2.0` | validation | 56.75% | -15.87% | 28 | 42.86% |
| previous baseline `1.0/2.0` | OOS extended | 13.92% | -29.38% | 39 | 38.46% |
| previous baseline `1.0/2.0` | OOS Q1 2026 | 41.98% | -21.03% | 20 | 50.00% |
| val-stability `0.5/1.75` | validation | 57.76% | -11.29% | 28 | 42.86% |
| val-stability `0.5/1.75` | OOS extended | 12.20% | -25.06% | 39 | 38.46% |
| val-stability `0.5/1.75` | OOS Q1 2026 | 34.02% | -16.74% | 20 | 50.00% |

Validation monthly profile for the selected exact replay:

| month | PnL | MDD | trades | WR |
|---|---:|---:|---:|---:|
| 2025-10 | 24.14% | -4.09% | 8 | 62.50% |
| 2025-11 | 25.93% | -11.29% | 14 | 35.71% |
| 2025-12 | 0.92% | -5.47% | 6 | 33.33% |

## Interpretation

This selection improves SOL drawdown without OOS-based tuning:

- Validation MDD improves from `-15.87%` to `-11.29%`.
- OOS extended MDD improves from `-29.38%` to `-25.06%`.
- OOS Q1 MDD improves from `-21.03%` to `-16.74%`.
- OOS extended PnL gives up `1.72%p`, from `13.92%` to `12.20%`.
- OOS Q1 PnL gives up `7.96%p`, from `41.98%` to `34.02%`.

The extended OOS MDD is still slightly below a strict `-25%` promotion budget
by about `0.06%p`. Because OOS was already inspected after selection, this run
should not be followed by more OOS-driven threshold nudging. Any next attempt
must predeclare a stricter validation-only policy before opening OOS again.

## Commands

```bash
/home/llewyn/miniconda3/bin/conda run -n quant_ai python scripts/search_sol_omega4_6_1_val_stability_20260708.py
/home/llewyn/miniconda3/bin/conda run -n quant_ai python scripts/apply_final_scale_map_sol_20260707.py --fixed-long-scale 0.5 --fixed-short-scale 1.75 --duration-gate-threshold 0.0055208323 --out-dir tmp/causal_regen_20260516/sol_val_stability_exact_20260708 --device auto
```
