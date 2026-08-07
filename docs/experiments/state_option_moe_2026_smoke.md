# State Option MoE 2026

Status: `reject`

## Summary

This experiment implements `state_option_moe_2026`, a state-tokenized option selector that is structurally different from the current clean-base / MuZero-AZ / Lifecycle family.

## OOS Results

| Metric | Value |
|---|---:|
| PnL 1x | `344.225333%` |
| MDD 1x | `-5.809556%` |
| Trades/day | `11.423729` |
| Avg leverage | `1.500742` |
| PnL 2x | `87.709623%` |
| PnL 3x | `-33.330406%` |

## Selected Config

`cv0.4_co0.0_to0.00_p0.55_u-0.0100_c30_mt16`

## Gate

- Baseline PnL gate: `False`
- Baseline MDD gate: `True`
- Trades/day gate: `True`
- Cost 3x survival: `False`
- Invariant audit: `True`

## Artifacts

- Report: `data/ensemble/reports/state_option_moe_2026_smoke.json`
- Grid: `data/ensemble/reports/state_option_moe_2026_smoke_grid.csv`
- Ledger: `data/ensemble/reports/state_option_moe_2026_smoke_ledger.csv`
- Model dir: `data/ensemble/supervised/state_option_moe_2026_smoke`
