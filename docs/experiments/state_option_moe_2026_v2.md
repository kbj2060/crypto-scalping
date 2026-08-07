# State Option MoE 2026 V2

Status: `promote`

## Summary

`state_option_moe_2026_v2` expands the SOMoE option catalog, adds upside-aware utility, and validation-selects execution risk profiles without changing existing v1 artifacts.

## OOS Results

| Metric | Value |
|---|---:|
| PnL 1x | `4007.372308%` |
| MDD 1x | `-10.055092%` |
| Trades/day | `10.745763` |
| Avg leverage | `2.669432` |
| PnL 2x | `168.619738%` |
| PnL 3x | `-5.556414%` |

## Selected Config

`v2_cv0.00_co0.15_ma0.20_to0.10_up0.00_p0.80_u0.000_mt18_cap4.2_sc1.08_rp2`

## Gate

- Clean-base PnL gate: `True`
- Clean-base MDD gate: `True`
- V1 PnL lift gate: `True`
- Trades/day gate: `True`
- Cost 3x not worse than V1: `True`
- Invariant audit: `True`

## Artifacts

- Report: `/home/llewyn/crypto-scalping/data/ensemble/reports/state_option_moe_2026_v2.json`
- Grid: `/home/llewyn/crypto-scalping/data/ensemble/reports/state_option_moe_2026_v2_grid.csv`
- Ledger: `/home/llewyn/crypto-scalping/data/ensemble/reports/state_option_moe_2026_v2_ledger.csv`
- Model dir: `/home/llewyn/crypto-scalping/data/ensemble/supervised/state_option_moe_2026_v2`
