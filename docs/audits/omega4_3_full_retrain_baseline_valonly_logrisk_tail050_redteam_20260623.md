# Omega4.3 Full-Retrain Baseline Red-Team Audit - 2026-06-23

## Verdict

`omega4_3_full_retrain_baseline_valonly_logrisk_tail050_20260623`: **`REDTEAM_PASS_CLEAN_RESEARCH_TEST_NOT_UPGRADE`**

This run retrained the Omega4.3 parent 3-head/exit bundle with full train rows and full exit samples, then retrained the baseline HGB margin+leverage sidecar with the Omega4.3 validation-only log-risk tail050 contract. It is not an upgrade candidate.

## Parent-Only Readout

The freshly retrained parent q0.70 readout before ATR safety and risk sidecar was weak:

- Validation: `+4.01%`, MDD `-16.38%`, WR `40.00%`, trades `35`
- OOS: `+4.02%`, MDD `-9.27%`, WR `40.91%`, trades `22`

## Metrics

| Split | PnL | MDD | WR | Trades | Avg Notional | Avg Margin | Avg Lev | Log-Risk Utility |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Current baseline validation | `+30.33%` | `-7.91%` | `67.06%` | `85` | `0.5536` | `0.2346` | `2.2574` | `0.205548` |
| Full retrain sizing validation | `+4.83%` | `-6.94%` | `67.16%` | `134` | `0.2761` | `0.1459` | `1.6733` | `0.045278` |
| Current baseline OOS readout | `+32.44%` | `-5.72%` | `66.15%` | `65` | `0.5613` | `0.2374` | `2.2699` | `0.262865` |
| Full retrain sizing OOS readout | `-10.23%` | `-14.58%` | `65.75%` | `73` | `0.4140` | `0.1940` | `1.9569` | `-0.128059` |

## Full Replay Diagnostic

| Split | PnL | MDD | WR | Trades | Log-Risk Utility |
| --- | ---: | ---: | ---: | ---: | ---: |
| Current baseline validation full replay | `+31.34%` | `-10.52%` | `68.35%` | `79` | `0.191867` |
| Full retrain validation full replay | `-3.78%` | `-11.30%` | `66.41%` | `128` | `-0.055182` |
| Current baseline OOS full replay | `+33.73%` | `-5.73%` | `66.10%` | `59` | `0.271981` |
| Full retrain OOS full replay | `-8.90%` | `-13.29%` | `65.22%` | `69` | `-0.116540` |

## Selection Hygiene

- Selected variant: `risk_1683`
- Validation-only recomputed top: `risk_1683`
- OOS excluded from filter/sort/tie-break.
- Live wiring unchanged.

## Checks

- `parent_report_exists`: `True`

- `parent_bundle_exists`: `True`

- `risk_report_exists`: `True`

- `risk_sidecar_exists`: `True`

- `risk_sidecar_loads`: `True`

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

## Upgrade Checks

- `validation_pnl_improved_vs_baseline`: `False`

- `validation_utility_improved_vs_baseline`: `False`

- `oos_pnl_improved_vs_baseline_readout`: `False`

- `oos_utility_improved_vs_baseline_readout`: `False`

- `full_replay_validation_pnl_improved_vs_baseline`: `False`

- `full_replay_oos_pnl_improved_vs_baseline`: `False`

## Artifacts

- Parent report: `tmp/causal_regen_20260516/omega4_3head_parent72_loose_entry_quality_20260620_full_retrain_baseline_e2_fulltrain_fullexit_q070_20260623/report.json`
- Parent bundle: `tmp/causal_regen_20260516/omega4_3head_parent72_loose_entry_quality_20260620_full_retrain_baseline_e2_fulltrain_fullexit_q070_20260623/true_3head_tabm_bundle.pt`
- Risk report: `tmp/causal_regen_20260516/omega4_2_trade_risk_sidecar_20260622_v9_full_retrain_baseline_parent_e2_fulltrain_fullexit_valonly_logrisk_tail050_20260623/report.json`
- Risk sidecar: `tmp/causal_regen_20260516/omega4_2_trade_risk_sidecar_20260622_v9_full_retrain_baseline_parent_e2_fulltrain_fullexit_valonly_logrisk_tail050_20260623/risk_sidecar.pkl`
- JSON audit: `tmp/causal_regen_20260516/omega4_3_full_retrain_baseline_valonly_logrisk_tail050_redteam_20260623/report.json`
