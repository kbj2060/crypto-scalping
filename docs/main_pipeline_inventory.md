# Main Pipeline Inventory

Current main stack:

- M7 data: improved M7 redesign clean dataset.
- DSAC: retrained on improved M7 clean data.
- Canonical live checkpoint: `data/ensemble/ckpt/best_dsac_agents.pth`.
- Named best checkpoint: `data/ensemble/ckpt/best_dsac_agents_redesign_clean_legacy.pth`.
- 2026 OOS report: `data/ensemble/reports/eval_2026_redesign_clean_legacy.json`.
- Backup root for archived artifacts: `backups/main_redesign_clean_legacy_20260423_220352`.

## Core Code To Keep

Data and feature pipeline:

- `core/data_collector.py`
- `core/feature_engineering.py`
- `features/engineering.py`
- `features/elite.py`
- `features/high_order_state.py`
- `features/m7.py`
- `features/registry.py`
- `features/schema.py`
- `pipeline/feature_contract.py`
- `pipeline/build_rl_dataset.py`
- `pipeline/augment_m7_dataset.py`
- `pipeline/run_train.py`
- `docs/unified_pipeline_design.md`
- `docs/feature_contract_manifest.json`

M7 supervised and unsupervised training/inference:

- `ensemble/seven_model_ensemble.py`
- `ensemble/artifact_utils.py`
- `ensemble/optuna_helper.py`
- `ensemble/supervised/common.py`
- `ensemble/supervised/live_supervised_hub.py`
- `ensemble/supervised/train_trend_xgb.py`
- `ensemble/supervised/train_entry_price_model.py`
- `ensemble/supervised/train_quantile_forest.py`
- `ensemble/supervised/train_multitarget_lgbm.py`
- `ensemble/supervised/train_manifold_hgb.py`
- `ensemble/unsupervised/common.py`
- `ensemble/unsupervised/live_unsupervised_hub.py`
- `ensemble/unsupervised/train_gmm_volatility.py`
- `ensemble/unsupervised/train_isolation_forest.py`
- `ensemble/unsupervised/train_vae_anomaly.py`

DSAC training, runtime, and validation:

- `ensemble/rl_continuous_common.py`
- `ensemble/rl_runtime_primitives.py`
- `ensemble/train_rl_dsac_agent.py`
- `scripts/eval_2026_oos.py`
- `scripts/analyze_dsac_state_ablation.py`
- `trading_bot.py`

## Code Removal Candidates

These files are not part of the current main path. Keep them only if their specific experiment is still needed.

- `ensemble/train_rl_sac_agent.py`
- `ensemble/train_rl_meta_agent.py`
- `ensemble/train_rl_meta_gating.py`
- `ensemble/msaf_formula.py`
- `ensemble/ensemble_router.py`
- `ensemble/diagnose_side_bias.py`
- `ensemble/unsupervised/train_hdbscan_regime.py`
- `scripts/ab_meta_gating_stages.py`
- `scripts/ab_test_dsac_balanced.py`
- `scripts/ab_test_dsac_sttp.py`
- `scripts/backtest_api_limit_30d.py`
- `scripts/backtest_dsac_execution_overlay_oos.py`
- `scripts/backtest_fraction_leverage_split.py`
- `scripts/backtest_limit_idea_suite.py`
- `scripts/backtest_live_limit_realtime.py`
- `scripts/backtest_macro_micro_playbook.py`
- `scripts/backtest_msaf_formula.py`
- `scripts/backtest_onchain_micro_core.py`
- `scripts/backtest_param_ensemble.py`
- `scripts/backtest_playbook_microstructure_architecture.py`
- `scripts/backtest_polymarket_event_follow_defense.py`
- `scripts/backtest_polymarket_exit_modes.py`
- `scripts/backtest_polymarket_exit_recovery_opt.py`
- `scripts/backtest_polymarket_shock_entry_exit.py`
- `scripts/backtest_polymarket_veto_panic.py`
- `scripts/backtest_quant_formulas_2025_rl.py`
- `scripts/backtest_replay_engine_kelly_leverage.py`
- `scripts/backtest_rl2025_native_formula.py`
- `scripts/backtest_top5_playbook_formula.py`
- `scripts/backtest_trading_bot_kelly_leverage.py`
- `scripts/backtest_ultimate_3model_ensemble.py`
- `scripts/compare_dsac_simple_models.py`
- `scripts/compare_dsac_split_ab.py`
- `scripts/compare_dsac_trendxgb_combo.py`
- `scripts/compare_dsac_trendxgb_proper.py`
- `scripts/compare_entry_exit_thresholds.py`
- `scripts/custom_rule_experiment.py`
- `scripts/eval_2026_dsac_limit.py`
- `scripts/eval_2026_oos_limit_overlay.py`
- `scripts/eval_primary_fill_2026.py`
- `scripts/evaluate_dsac_closedloop_sttp.py`
- `scripts/experiment_enhanced_module_sweep.py`
- `scripts/explore_replay_reweight_policies.py`
- `scripts/explore_trade_level_fraction_leverage.py`
- `scripts/explore_unsup_redesign_m7.py`
- `scripts/fine_tune_dsac_duckdb_30m.py`
- `scripts/fine_tune_dsac_duckdb_polymarket.py`
- `scripts/fine_tune_dsac_polymarket_api_max.py`
- `scripts/optimize_duckdb_quant_formula.py`
- `scripts/optimize_polymarket_leading_indicator.py`
- `scripts/run_blueprint_ab_template.py`
- `scripts/run_m7_ablation4_backtest.py`
- `scripts/search_m7_configs.py`
- `scripts/tune_fuse_params.py`
- `scripts/tune_proposed_backtest_params.py`
- `scripts/tune_static_and_regime_dual_ensemble.py`

## Notes

- Do not delete the backup folder until the bot has run stably with the canonical checkpoint for a while.
- `best_dsac_agents.pth` is now intentionally the same model as `best_dsac_agents_redesign_clean_legacy.pth`.
- HDBSCAN artifacts were archived because the current M7 redesign does not use `hdbscan_regime`.
- Long/Short specialist display has been removed from the active live path.
- `OnlineHMMDetector` and `MultiTimeframeFeatures` now live in `ensemble/rl_runtime_primitives.py`.
