# State Option MoE 2026

Status: `reject`

## Summary

This experiment implements `state_option_moe_2026`, a state-tokenized option selector that is structurally different from the current clean-base / MuZero-AZ / Lifecycle family.

## OOS Results

| Metric | Value |
|---|---:|
| PnL 1x | `362.656101%` |
| MDD 1x | `-10.073051%` |
| Trades/day | `5.254237` |
| Avg leverage | `2.113226` |
| PnL 2x | `11.151300%` |
| PnL 3x | `-26.855081%` |

## Selected Config

`cv0.0_co0.0_to0.00_p0.55_u0.0000_c30_mt12`

## Gate

- Baseline PnL gate: `False`
- Baseline MDD gate: `True`
- Trades/day gate: `False`
- Cost 3x survival: `False`
- Invariant audit: `True`

## Artifacts

- Report: `/home/llewyn/crypto-scalping/data/ensemble/reports/state_option_moe_2026.json`
- Grid: `/home/llewyn/crypto-scalping/data/ensemble/reports/state_option_moe_2026_grid.csv`
- Ledger: `/home/llewyn/crypto-scalping/data/ensemble/reports/state_option_moe_2026_ledger.csv`
- Model dir: `/home/llewyn/crypto-scalping/data/ensemble/supervised/state_option_moe_2026`
