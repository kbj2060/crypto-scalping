# State Option MoE 2026 V2

Status: `reject`

## Summary

`state_option_moe_2026_v2` expands the SOMoE option catalog, adds upside-aware utility, and validation-selects execution risk profiles without changing existing v1 artifacts.

## OOS Results

| Metric | Value |
|---|---:|
| PnL 1x | `-32.322168%` |
| MDD 1x | `-39.502679%` |
| Trades/day | `4.542373` |
| Avg leverage | `1.071418` |
| PnL 2x | `-53.941309%` |
| PnL 3x | `-64.461344%` |

## Selected Config

`v2_cv0.25_co0.15_ma0.00_to0.10_up0.30_p0.80_u-0.004_mt12_cap3.6_sc1.00_rp0`

## Gate

- Clean-base PnL gate: `False`
- Clean-base MDD gate: `False`
- V1 PnL lift gate: `False`
- Trades/day gate: `False`
- Cost 3x not worse than V1: `False`
- Invariant audit: `True`

## Artifacts

- Report: `data/ensemble/reports/state_option_moe_2026_v2_noleak.json`
- Grid: `data/ensemble/reports/state_option_moe_2026_v2_noleak_grid.csv`
- Ledger: `data/ensemble/reports/state_option_moe_2026_v2_noleak_ledger.csv`
- Model dir: `data/ensemble/supervised/state_option_moe_2026_v2_noleak`
