# clean_base_plus_conviction_sleeve_challenger_v1

## Summary

Deterministic validation-selected Conviction Sleeve challenger over frozen Lifecycle V1 core.

## OOS Metrics

- PnL 1x: 207.236888
- MDD 1x: -18.016318
- Cost2 PnL: 133.046834
- Cost3 PnL: -8.649827
- Sleeve fraction: 0.000000
- Mode counts: {"NO_SLEEVE": 363, "ADD_SAME_SIDE_15": 0, "ADD_SAME_SIDE_25": 0, "HEDGE_OPPOSITE_15": 0, "HEDGE_OPPOSITE_25": 0}

## Verdict

- reject_for_promotion_gate
- Reject reasons: total PnL >= 230, total MDD >= -18.0, cost3 >= 70

Cost stress rebuilds multiplier-specific entry and exit slippage contexts.

## Artifacts

- Report: `/home/llewyn/crypto-scalping/data/ensemble/reports/clean_base_plus_conviction_sleeve_challenger_v1_2026.json`
- Grid: `/home/llewyn/crypto-scalping/data/ensemble/reports/clean_base_plus_conviction_sleeve_challenger_v1_grid.csv`
- Ledger: `/home/llewyn/crypto-scalping/data/ensemble/reports/clean_base_plus_conviction_sleeve_challenger_v1_ledger.csv`
- Model: `/home/llewyn/crypto-scalping/data/ensemble/supervised/clean_base_plus_conviction_sleeve_challenger_v1/conviction_sleeve_policy_grid.pkl`
