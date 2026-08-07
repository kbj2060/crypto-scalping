# Omega4.4 v18 Red-Team Audit

- Verdict: `REDTEAM_PASS_FULL_PROMOTABLE`
- Research reproduction pass: `True`
- Promotion red-team pass: `True`
- Source report: `/home/llewyn/crypto-scalping/tmp/causal_regen_20260516/omega4_2_trade_risk_sidecar_20260622_v18_topdown_best_parent_exit075_live_exposure_dynamic_leverage_valonly_logrisk_tail050_minavg075_20260624/report.json`
- Audit JSON: `/home/llewyn/crypto-scalping/tmp/causal_regen_20260516/omega4_4_v18_redteam_audit_20260625/report.json`

## Metrics

| Split | PnL | MDD | Trades | WR | Avg notional | Avg leverage |
|---|---:|---:|---:|---:|---:|---:|
| sizing_only_validation | 39.4334 | -11.1220 | 44 | 0.5909 | 0.7682 | 2.1458 |
| sizing_only_oos | 36.6465 | -10.7378 | 36 | 0.6667 | 0.7916 | 2.1653 |
| full_replay_validation | 35.8538 | -14.1696 | 41 | 0.5366 | 0.7595 | 2.1397 |
| full_replay_oos | 43.2312 | -10.7585 | 34 | 0.6765 | 0.7931 | 2.1654 |

## Blocking Result

- No hard reproduction failures.

## Key Passes

- PASS `artifact_presence`: missing=[]
- PASS `risk_sidecar_loads`: keys=['contract', 'dynamic_leverage', 'feature_columns', 'log_risk_params', 'model', 'model_kind', 'notional_scaled_sltp', 'risk_feature_mode', 'score_quality_blend', 'selected_mapping', 'selection_objective', 'selection_scope', 'side_split_model', 'target_mae_penalty', 'train_score_iqr', 'train_score_q50']
- PASS `feature_columns_present`: count=29
- PASS `forbidden_feature_hits_zero`: hits=[]
- PASS `quality_threshold_070`: value=True
- PASS `exit_threshold_075`: value=True
- PASS `model_kind_hgb`: value=True
- PASS `feature_mode_parent_outputs`: value=True
- PASS `side_split_enabled`: value=True
- PASS `dynamic_leverage_enabled`: value=True
- PASS `selection_scope_validation_only`: value=True
- PASS `selection_objective_log_risk`: value=True
- PASS `tail_penalty_050`: value=True
- PASS `target_mae_penalty_050`: value=True
- PASS `notional_scaled_sltp_false`: value=True
- PASS `notional_contract_declared`: value=True
- PASS `selected_mapping_top_validation_only_eligible`: {"eligible_count": 1916, "rank": 1, "selected_variant": "risk_1700", "top_validation_log_risk_utility": 0.19815665903976526, "top_validation_pnl": 39.43335549560692, "top_variant": "risk_1700", "trade_floor": 41}
- PASS `ledger_notional_math_exact`: max_error=0.0
- PASS `validation_full_replay_positive`: {"pnl": 35.853831530265, "mdd": -14.16964911682127, "trades": 41, "wr": 0.5365853658536586, "avg_notional": 0.7594724821298097, "avg_margin_fraction": 0.35319800326308676, "avg_leverage": 2.139690893122123, "log_risk_utility": 0.16937285202190996, "exit_reasons": {"stop_loss": 18, "take_profit": 12, "exit_head": 10, "forced_end": 1}}
- PASS `oos_full_replay_positive`: {"pnl": 43.2312455386217, "mdd": -10.758485734455137, "trades": 34, "wr": 0.6764705882352942, "avg_notional": 0.7931127953001704, "avg_margin_fraction": 0.3644782477053848, "avg_leverage": 2.165355179920918, "log_risk_utility": 0.29053849141535804, "exit_reasons": {"stop_loss": 8, "take_profit": 4, "exit_head": 21, "forced_end": 1}}
- PASS `validation_full_replay_mdd_within_16pct`: mdd=-14.16964911682127
- PASS `oos_full_replay_mdd_within_16pct`: mdd=-10.758485734455137
- PASS `standalone_promotion_manifest_exists`: exists=True path=/home/llewyn/crypto-scalping/tmp/causal_regen_20260516/omega4_2_trade_risk_sidecar_20260622_v18_topdown_best_parent_exit075_live_exposure_dynamic_leverage_valonly_logrisk_tail050_minavg075_20260624/promotion_manifest.json
- PASS `standalone_runtime_contract_exists`: exists=True path=/home/llewyn/crypto-scalping/tmp/causal_regen_20260516/omega4_2_trade_risk_sidecar_20260622_v18_topdown_best_parent_exit075_live_exposure_dynamic_leverage_valonly_logrisk_tail050_minavg075_20260624/runtime_contract.json
- PASS `candidate_manifest_exists`: exists=True path=/home/llewyn/crypto-scalping/data/ensemble/supervised/omega4_4_v18_baseline_20260624/candidate_manifest.json
- PASS `contract_doc_exists`: exists=True path=/home/llewyn/crypto-scalping/docs/model_contracts/omega4_4_v18_baseline_20260624_contract.md
- PASS `promotion_manifest_model_id_unique_v18`: model_id=omega4_4_v18_baseline_20260624
- PASS `promotion_manifest_source_model_id_preserved`: source_report_model_id=omega4_2_trade_risk_sidecar_20260622 report_model_id=omega4_2_trade_risk_sidecar_20260622
- PASS `runtime_contract_model_id_unique_v18`: model_id=omega4_4_v18_baseline_20260624
- PASS `runtime_contract_full_replay_enabled`: execution_contract={'parent_model_owns': ['direction', 'quality_gate', 'exit_head', 'entry_time_atr_sltp_barrier_timing'], 'risk_sidecar_owns': ['entry_time_margin_fraction', 'entry_time_leverage', 'trade_pnl_sizing'], 'full_replay_dynamic_exit_enabled': True, 'exit_sizing_input_mode': 'actual', 'runtime_must_fail_on_missing_sidecar_or_contract_mismatch': True}
- PASS `runtime_contract_fail_fast_required`: fail_fast_required=True
- PASS `candidate_manifest_model_id_unique_v18`: model_id=omega4_4_v18_baseline_20260624
- PASS `candidate_manifest_runtime_contract_ref`: runtime_contract=tmp/causal_regen_20260516/omega4_2_trade_risk_sidecar_20260622_v18_topdown_best_parent_exit075_live_exposure_dynamic_leverage_valonly_logrisk_tail050_minavg075_20260624/runtime_contract.json
- PASS `full_replay_candidate_is_diagnostic_single_candidate`: count=1 variants=['risk_1700']
