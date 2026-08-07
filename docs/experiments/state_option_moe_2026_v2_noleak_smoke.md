# State Option MoE 2026 V2

Status: `reject`

## Summary

`state_option_moe_2026_v2` expands the SOMoE option catalog, adds upside-aware utility, and validation-selects execution risk profiles without changing existing v1 artifacts.

## OOS Results

| Metric | Value |
|---|---:|
| PnL 1x | `0.203536%` |
| MDD 1x | `-4.113874%` |
| Trades/day | `0.457627` |
| Avg leverage | `1.000000` |
| PnL 2x | `-3.549001%` |
| PnL 3x | `-7.954415%` |

## Selected Config

`v2_cv0.00_co0.15_ma0.20_to0.10_up0.00_p0.80_u-0.004_mt18_cap3.6_sc1.00_rp1`

## Gate

- Clean-base PnL gate: `False`
- Clean-base MDD gate: `True`
- V1 PnL lift gate: `False`
- Trades/day gate: `False`
- Cost 3x not worse than V1: `True`
- Invariant audit: `True`

## Artifacts

- Report: `data/ensemble/reports/state_option_moe_2026_v2_noleak_smoke.json`
- Grid: `data/ensemble/reports/state_option_moe_2026_v2_noleak_smoke_grid.csv`
- Ledger: `data/ensemble/reports/state_option_moe_2026_v2_noleak_smoke_ledger.csv`
- Model dir: `data/ensemble/supervised/state_option_moe_2026_v2_noleak_smoke`
