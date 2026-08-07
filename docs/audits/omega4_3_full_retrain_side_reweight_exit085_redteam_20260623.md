# Omega4.3 Full-Retrain Side-Reweight Exit0.85 Audit - 2026-06-23

## Verdict

`omega4_3_full_retrain_side_reweight_exit085_valonly_logrisk_tail050_20260623`: **`REDTEAM_PASS_CLEAN_RESEARCH_IMPROVED_NOT_CURRENT_UPGRADE`**

This candidate improves the failed full-retrain baseline by adding side-aware class weights to the parent training loss and raising the runtime exit threshold to `0.85`. It is still not a replacement for the current Omega4.3 baseline.

## What Changed

- Parent full train/full exit kept enabled.
- Direction and quality class weights: cash `1.00`, long `0.65`, short `1.35`.
- Exit threshold: `0.70 -> 0.85`.
- Risk sidecar kept baseline contract: HGB, side-split, parent-output features, dynamic leverage, validation-only log-risk, `tail_penalty=0.5`.

## Metrics

| Split | PnL | MDD | WR | Trades | Avg Notional | Avg Margin | Avg Lev | Log-Risk Utility |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Failed full retrain validation | `+4.83%` | `-6.94%` | `67.16%` | `134` | `0.2761` | `0.1459` | `1.6733` | `0.045278` |
| Improved candidate validation | `+12.23%` | `-9.26%` | `78.22%` | `101` | `0.4350` | `0.3021` | `1.4308` | `0.105724` |
| Current baseline validation | `+30.33%` | `-7.91%` | `67.06%` | `85` | `0.5536` | `0.2346` | `2.2574` | `0.205548` |
| Failed full retrain OOS | `-10.23%` | `-14.58%` | `65.75%` | `73` | `0.4140` | `0.1940` | `1.9569` | `-0.128059` |
| Improved candidate OOS | `+13.48%` | `-5.33%` | `81.18%` | `85` | `0.4560` | `0.3101` | `1.4642` | `0.121352` |
| Current baseline OOS | `+32.44%` | `-5.72%` | `66.15%` | `65` | `0.5613` | `0.2374` | `2.2699` | `0.262865` |

## Full Replay Diagnostic

| Split | PnL | MDD | WR | Trades | Log-Risk Utility |
| --- | ---: | ---: | ---: | ---: | ---: |
| Failed full retrain validation replay | `-3.78%` | `-11.30%` | `66.41%` | `128` | `-0.055182` |
| Improved validation replay | `+12.82%` | `-9.33%` | `77.78%` | `99` | `0.111008` |
| Current baseline validation replay | `+31.34%` | `-10.52%` | `68.35%` | `79` | `0.191867` |
| Failed full retrain OOS replay | `-8.90%` | `-13.29%` | `65.22%` | `69` | `-0.116540` |
| Improved OOS replay | `+13.45%` | `-5.34%` | `80.72%` | `83` | `0.121032` |
| Current baseline OOS replay | `+33.73%` | `-5.73%` | `66.10%` | `59` | `0.271981` |

## Checks

- `parent_report_exists`: `True`

- `parent_bundle_exists`: `True`

- `risk_report_exists`: `True`

- `risk_sidecar_exists`: `True`

- `risk_sidecar_loads`: `True`

- `class_weights_recorded`: `True`

- `exit_threshold_085`: `True`

- `model_kind_is_hgb`: `True`

- `risk_feature_mode_parent_outputs`: `True`

- `side_split_model_enabled`: `True`

- `dynamic_leverage_enabled`: `True`

- `selection_scope_validation_only`: `True`

- `log_tail_penalty_050`: `True`

- `target_mae_penalty_050`: `True`

- `validation_only_top_matches_selected`: `True`

- `full_replay_selection_not_applied`: `True`

- `notional_contract_declared`: `True`

- `forbidden_feature_hits_zero`: `True`

## Improvement Checks

- `validation_pnl_improved_vs_failed_full_retrain`: `True`

- `oos_pnl_improved_vs_failed_full_retrain`: `True`

- `full_replay_validation_improved_vs_failed_full_retrain`: `True`

- `full_replay_oos_improved_vs_failed_full_retrain`: `True`

## Current Baseline Upgrade Checks

- `validation_pnl_improved_vs_current_baseline`: `False`

- `validation_utility_improved_vs_current_baseline`: `False`

- `oos_pnl_improved_vs_current_baseline`: `False`

- `oos_utility_improved_vs_current_baseline`: `False`

- `full_replay_validation_pnl_improved_vs_current_baseline`: `False`

- `full_replay_oos_pnl_improved_vs_current_baseline`: `False`

## Notes

- The attempted exit0.95 side-split risk run failed fast because long-side risk samples were only `6`.
- Live wiring unchanged.

## Artifacts

- Parent report: `tmp/causal_regen_20260516/omega4_3head_parent72_loose_entry_quality_20260620_full_retrain_side_reweight_l065_s135_e2_fulltrain_fullexit_q070_20260623/report.json`
- Risk report: `tmp/causal_regen_20260516/omega4_2_trade_risk_sidecar_20260622_v11_full_retrain_side_reweight_l065_s135_exit085_valonly_logrisk_tail050_20260623/report.json`
- Risk sidecar: `tmp/causal_regen_20260516/omega4_2_trade_risk_sidecar_20260622_v11_full_retrain_side_reweight_l065_s135_exit085_valonly_logrisk_tail050_20260623/risk_sidecar.pkl`
- JSON audit: `tmp/causal_regen_20260516/omega4_3_full_retrain_side_reweight_exit085_redteam_20260623/report.json`
