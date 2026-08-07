# clean_base_plus_causal_conviction_sleeve_v1_1

## Summary

Causal sleeve scorer using train-only future labels and runtime predicted utility only.

## OOS Metrics
- PnL 1x: 210.491277
- MDD 1x: -18.015155
- Cost2 PnL: 133.150031
- Cost3 PnL: -8.953223
- Sleeve fraction: 0.170799
- Actions: {"NO_SLEEVE": 301, "ADD_SAME_SIDE_15": 0, "ADD_SAME_SIDE_25": 62, "HEDGE_OPPOSITE_15": 0, "HEDGE_OPPOSITE_25": 0}

## Verdict
- reject_for_promotion_gate
- Reject reasons: total_pnl >= 230, total_mdd >= -18.0, cost3 >= 70

Runtime sleeve decisions do not use future realized prices.

## Artifacts
- Report: `/home/llewyn/crypto-scalping/data/ensemble/reports/clean_base_causal_sleeve_mdd_guard_v1_3_2026.json`
- Grid: `/home/llewyn/crypto-scalping/data/ensemble/reports/clean_base_causal_sleeve_mdd_guard_v1_3_grid.csv`
- Ledger: `/home/llewyn/crypto-scalping/data/ensemble/reports/clean_base_causal_sleeve_mdd_guard_v1_3_ledger.csv`
- Model: `/home/llewyn/crypto-scalping/data/ensemble/supervised/clean_base_causal_sleeve_mdd_guard_v1_3/causal_sleeve_regressors.pkl`
