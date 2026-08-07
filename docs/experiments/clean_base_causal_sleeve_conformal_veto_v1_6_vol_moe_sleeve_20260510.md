# Clean Base Causal Sleeve Conformal Veto V1.6 Volatility-MoE Sleeve

Status: `reject`

## OOS Metrics

| Metric | Value |
|---|---:|
| PnL 1x | `-1.684615%` |
| MDD 1x | `-2.781135%` |
| Trades/day | `1.186441` |
| Sleeve fraction | `0.000000` |
| Cost2 PnL | `-22.806374%` |
| Cost3 PnL | `-10.991548%` |

## Selected Config

`sv_same0.0015_frac0.25_bars6_acct0.08_day0.015_q0.70_lcb-0.0060_adv0.020`

## Audit

- Preservation: `True`
- Accounting: `True`
- Causality: `True`

## MoE Training

`{"long_calm": {"rows": 87, "trained": false, "fallback": "global"}, "long_funding": {"rows": 11, "trained": false, "fallback": "global"}, "long_high_vol": {"rows": 27, "trained": false, "fallback": "global"}, "long_tail_liq": {"rows": 26, "trained": false, "fallback": "global"}, "short_calm": {"rows": 102, "same_positive_rate": 0.20588235294117646, "hedge_positive_rate": 0.1568627450980392, "same_mean": -0.0060453743810183315, "hedge_mean": -0.003971358155398884, "trained": true}, "short_funding": {"rows": 10, "trained": false, "fallback": "global"}, "short_high_vol": {"rows": 13, "trained": false, "fallback": "global"}, "short_tail_liq": {"rows": 30, "trained": false, "fallback": "global"}}`

## Artifacts

- Report: `/home/llewyn/crypto-scalping/data/ensemble/reports/clean_base_causal_sleeve_conformal_veto_v1_6_vol_moe_sleeve_20260510.json`
- Grid: `/home/llewyn/crypto-scalping/data/ensemble/reports/clean_base_causal_sleeve_conformal_veto_v1_6_vol_moe_sleeve_20260510_grid.csv`
- Ledger: `/home/llewyn/crypto-scalping/data/ensemble/reports/clean_base_causal_sleeve_conformal_veto_v1_6_vol_moe_sleeve_20260510_ledger.csv`
- Model: `/home/llewyn/crypto-scalping/data/ensemble/supervised/clean_base_causal_sleeve_conformal_veto_v1_6_vol_moe_sleeve_20260510/sleeve_conformal_veto.pkl`
