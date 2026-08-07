# State Option MoE 2026 V2

Status: `reject`

## Summary

`state_option_moe_2026_v2` expands the SOMoE option catalog, adds upside-aware utility, and validation-selects execution risk profiles without changing existing v1 artifacts.

## OOS Results

| Metric | Value |
|---|---:|
| PnL 1x | `23.458982%` |
| MDD 1x | `-22.118966%` |
| Trades/day | `4.932203` |
| Avg leverage | `1.580756` |
| PnL 2x | `13.404646%` |
| PnL 3x | `-29.618245%` |

## Selected Config

`v2_cv0.00_co0.00_ma0.20_to0.10_up0.15_p0.80_u-0.004_mt12_cap3.2_sc1.00_rp1`

## Gate

- Clean-base PnL gate: `False`
- Clean-base MDD gate: `False`
- V1 PnL lift gate: `False`
- Trades/day gate: `False`
- Cost 3x not worse than V1: `False`
- Invariant audit: `True`

## Artifacts

- Report: `data/ensemble/reports/state_option_moe_2026_v2_smoke.json`
- Grid: `data/ensemble/reports/state_option_moe_2026_v2_smoke_grid.csv`
- Ledger: `data/ensemble/reports/state_option_moe_2026_v2_smoke_ledger.csv`
- Model dir: `data/ensemble/supervised/state_option_moe_2026_v2_smoke`
