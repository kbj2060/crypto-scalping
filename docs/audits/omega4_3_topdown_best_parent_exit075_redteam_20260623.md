# Omega4.3 Top-Down Best Parent Exit0.75 Audit - 2026-06-23

## Verdict

`omega4_3_topdown_best_parent_exit075_valonly_logrisk_tail050_20260623`: **`REDTEAM_PASS_CLEAN_RESEARCH_TOPDOWN_IMPROVED_NOT_CURRENT_UPGRADE`**

This run follows the best-performing stack top-down: retrain the best-known parent recipe first, sweep exit threshold for that parent, then train the Omega4.3 HGB risk sidecar on the selected exit setting. It improves the failed full retrain materially, but it is not a current-baseline replacement.

## Why Full Train Was Worse

Full train changed the parent direction/quality surface. The failed full retrain had lower classification loss but shifted the trading distribution toward weaker long-heavy/excess-reentry behavior. The best historical recipe used a capped `15k` direction train window and `15k` exit samples, preserving the short-heavy edge that the full split diluted.

## Top-Down Sequence

1. Parent recipe: `same_as_direction`, terminal-giveback exit labels, `epochs=2`, `max_train_rows=15000`, `max_exit_samples=15000`, q0.70.
2. Exit sweep on the retrained parent selected `exit_threshold=0.75` by validation stability.
3. Risk sidecar: HGB, side-split, parent-output features, dynamic leverage, validation-only log-risk, `tail_penalty=0.5`.

Parent-only q0.70 readout:
- Validation `+26.09%`, MDD `-7.60%`, trades `35`
- OOS `+27.38%`, MDD `-4.90%`, trades `22`

## Metrics

| Split | PnL | MDD | WR | Trades | Avg Notional | Avg Margin | Avg Lev | Log-Risk Utility |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Failed full retrain validation | `+4.83%` | `-6.94%` | `67.16%` | `134` | `0.2761` | `0.1459` | `1.6733` | `0.045278` |
| Top-down candidate validation | `+19.10%` | `-6.97%` | `59.09%` | `44` | `0.4330` | `0.2389` | `1.7761` | `0.148370` |
| Current baseline validation | `+30.33%` | `-7.91%` | `67.06%` | `85` | `0.5536` | `0.2346` | `2.2574` | `0.205548` |
| Failed full retrain OOS | `-10.23%` | `-14.58%` | `65.75%` | `73` | `0.4140` | `0.1940` | `1.9569` | `-0.128059` |
| Top-down candidate OOS | `+22.21%` | `-5.55%` | `66.67%` | `36` | `0.4567` | `0.2471` | `1.8103` | `0.187005` |
| Current baseline OOS | `+32.44%` | `-5.72%` | `66.15%` | `65` | `0.5613` | `0.2374` | `2.2699` | `0.262865` |

## Full Replay Diagnostic

| Split | PnL | MDD | WR | Trades | Log-Risk Utility |
| --- | ---: | ---: | ---: | ---: | ---: |
| Failed full retrain validation replay | `-3.78%` | `-11.30%` | `66.41%` | `128` | `-0.055182` |
| Top-down validation replay | `+19.10%` | `-9.03%` | `59.09%` | `44` | `0.148365` |
| Current baseline validation replay | `+31.34%` | `-10.52%` | `68.35%` | `79` | `0.191867` |
| Failed full retrain OOS replay | `-8.90%` | `-13.29%` | `65.22%` | `69` | `-0.116540` |
| Top-down OOS replay | `+25.70%` | `-5.56%` | `68.57%` | `35` | `0.215091` |
| Current baseline OOS replay | `+33.73%` | `-5.73%` | `66.10%` | `59` | `0.271981` |

## Checks

- `parent_report_exists`: `True`

- `parent_bundle_exists`: `True`

- `parent_recipe_train15k_exit15k`: `True`

- `risk_report_exists`: `True`

- `risk_sidecar_exists`: `True`

- `risk_sidecar_loads`: `True`

- `exit_threshold_075`: `True`

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

## Artifacts

- Parent report: `tmp/causal_regen_20260516/omega4_3head_parent72_loose_entry_quality_20260620_topdown_best_parent_e2_train15k_exit15k_q070_20260623/report.json`
- Risk report: `tmp/causal_regen_20260516/omega4_2_trade_risk_sidecar_20260622_v14_topdown_best_parent_e2_train15k_exit15k_exit075_valonly_logrisk_tail050_20260623/report.json`
- Risk sidecar: `tmp/causal_regen_20260516/omega4_2_trade_risk_sidecar_20260622_v14_topdown_best_parent_e2_train15k_exit15k_exit075_valonly_logrisk_tail050_20260623/risk_sidecar.pkl`
- JSON audit: `tmp/causal_regen_20260516/omega4_3_topdown_best_parent_exit075_redteam_20260623/report.json`
