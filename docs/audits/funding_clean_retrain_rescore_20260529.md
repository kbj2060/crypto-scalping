# Funding-Clean Derived Artifact Retrain/Rescore - 2026-05-29

## Summary

`last_funding_rate` contamination remediation is now extended beyond direct CSV patching. The active split CSVs were already regenerated to ETHUSDT-only backward-asof funding; this run retrained or rescored the main derived artifacts that consumed funding-family inputs.

This matters because direct CSV patching alone does not clean frozen model outputs. Any artifact trained or scored before the funding fix can still embed old ETHFIUSDT or future-filled funding behavior.

## Root Bug

- Historical year split files had contaminated `last_funding_rate`.
- 2024 split mostly used future ETHUSDT funding by front-fill.
- 2025/2026 splits mostly used future ETHFIUSDT funding.
- Correct contract is ETHUSDT funding only, aligned by backward/as-of timestamp merge.

Primary bug audit:

- `docs/audits/last_funding_rate_source_audit_20260528.md`

## Run Directory

- `tmp/causal_regen_20260516/funding_clean_retrain_20260529`

## M7 Retrain And Rescore

Active M7 artifacts were backed up before replacement:

- `tmp/causal_regen_20260516/funding_clean_retrain_20260529/backup_active_m7_before_replace`

Retrained and replaced active artifacts:

- `data/ensemble/supervised/entry_price_model.json`
- `data/ensemble/supervised/entry_price_model.pkl`
- `data/ensemble/supervised/trend_xgb.json`
- `data/ensemble/supervised/trend_xgb.pkl`
- `data/ensemble/supervised/multi_target_lgbm.json`
- `data/ensemble/supervised/multi_target_lgbm.pkl`
- `data/ensemble/supervised/quantile_forest.json`
- `data/ensemble/supervised/quantile_forest.pkl`
- `data/ensemble/unsupervised/vae_anomaly.json`
- `data/ensemble/unsupervised/vae_anomaly.pkl`

Retraining log:

- `tmp/causal_regen_20260516/funding_clean_retrain_20260529/logs/m7_retrain.log`

Key metrics:

- Entry price model: `val_long_mae=0.001249`, `val_short_mae=0.001210`, `test_long_mae=0.001814`, `test_short_mae=0.001664`
- Trend XGB/LGBM route: GPU fallback to CPU; best validation directional F1 about `0.7878`, test balanced accuracy `0.5367`
- Multi-target LGBM: `dir_bal_acc=0.5310`, `quality_mae=0.003883`, `hold_mae=3.3836`
- Quantile forest: `mae=0.005522`, `dir_acc=0.4906`, `interval_width=0.010393`
- VAE anomaly: CUDA used; `threshold=0.186550`, `val_anomaly_ratio=0.0965`

Rescored M7 CSVs:

- `data/splits/year_oos/rl_training_2025_m7.csv`
- `data/splits/year_oos/rl_training_2026_m7_supervised_redesign_clean.csv`
- `tmp/causal_regen_20260516/splits/rl_training_2025_m7.csv`
- `tmp/causal_regen_20260516/splits/rl_training_2026_m7.csv`

Row counts:

- 2025 M7: `105064`
- 2026 M7: `16897`

Funding validation:

- 2025 M7 vs clean feature frame `last_funding_rate`: max abs diff `0.0`
- 2026 M7 vs clean feature frame `last_funding_rate`: max abs diff `0.0`

## Regime4 Future Predictor Rescore

The active `regime4_pred_tft_h12_nomdjd_all74_20260517` contract was preserved:

- horizon: `12`
- excluded features: `pred_mdjd`, `conf_mdjd`
- selected feature count: `74`
- selection policy: `drop_importance_below_0_exclude_pred_mdjd_conf_mdjd`

Active sidecar directory:

- `data/ensemble/supervised/regime4_pred_tft_h12_nomdjd_all74_20260517`

Backup before replacement:

- `tmp/causal_regen_20260516/funding_clean_retrain_20260529/backup_regime4_pred_before_replace`

Updated sidecars:

- `data/ensemble/supervised/regime4_pred_tft_h12_nomdjd_all74_20260517/training_features_2024_regime4_pred_tft_vsn_selected.csv`
- `data/ensemble/supervised/regime4_pred_tft_h12_nomdjd_all74_20260517/training_features_2025_regime4_pred_tft_vsn_selected.csv`
- `data/ensemble/supervised/regime4_pred_tft_h12_nomdjd_all74_20260517/training_features_2026_rebuilt_regime4_pred_tft_vsn_selected.csv`

Reports:

- `data/ensemble/reports/regime4_pred_tft_h12_nomdjd_all74_20260529_cleanfunding_report.json`
- `data/ensemble/reports/regime4_transform_2026_h12_nomdjd_all74_20260529_cleanfunding.json`

Key validation:

- 2025 prediction counts: `bull=31351`, `bear=24966`, `chop=26854`, `whipsaw=21930`
- 2025 confidence mean: `0.6219208135`
- selected validation balanced accuracy: `0.5787905595`
- 2026 rows: `16897`
- 2026 class probability sum min/max: approximately `1.0 / 1.0`
- 2026 prediction counts: `bull=5001`, `bear=3252`, `chop=3785`, `whipsaw=4859`

## A5Dir / Router Regeneration

Clean unified RL datasets were rebuilt before router regeneration so stale M7/AI columns were not reused.

Unified clean datasets:

- `tmp/causal_regen_20260516/funding_clean_retrain_20260529/rl_training_2024_unified_cleanfunding.csv`
- `tmp/causal_regen_20260516/funding_clean_retrain_20260529/rl_training_2025_unified_cleanfunding.csv`

Row counts:

- 2024 unified: `105380`
- 2025 unified: `105064`

A5Dir/router output:

- `tmp/causal_regen_20260516/alpha5_a5dir_2024_train_2025_score_20260529_cleanfunding_stable48`

Score CSV:

- `tmp/causal_regen_20260516/alpha5_a5dir_2024_train_2025_score_20260529_cleanfunding_stable48/08_alpha5_direction_router_rl_2024_to_2025/rl_training_2025_direction_router.csv`

Manifest:

- `tmp/causal_regen_20260516/alpha5_a5dir_2024_train_2025_score_20260529_cleanfunding_stable48/build_manifest.json`

Router feature contract:

- `tmp/causal_regen_20260516/alpha5_router5_full_candidate_search_20260521/rank_pruned_stable_top48_feature_list.json`

Router metrics:

- validation balanced accuracy: `0.6268177783`
- OOS balanced accuracy: `0.5490030238`

Scored 2025 router stats:

- rows: `105064`
- `a5dir_available_ratio=0.850691`
- `a5dir_long_prob_mean=0.2469517`
- `a5dir_short_prob_mean=0.2280394`
- `a5dir_none_prob_mean=0.5250090`
- `a5dir` probability sum min/max: `0.99999989 / 1.00000014`

Funding validation:

- A5Dir 2025 score CSV vs clean feature frame `last_funding_rate`: max abs diff `0.0`

## Execution Notes

The first A5Dir run with `--force` failed because the builder reused the default stale checkpoint directory `data/tmp/unified_build_ckpt_2024`, whose checkpoint lacked the current AI columns. This was not accepted as an active compatibility workaround. The run was rerun without `--force` using the explicitly rebuilt clean unified CSVs, so downstream label/router steps completed from clean inputs.

## Current Contract Status

Clean funding remediation is now complete for:

- active split feature CSVs,
- direct RL base/training CSV funding columns,
- active M7 artifacts and M7-scored CSVs,
- active `regime4_pred` h12 all74 sidecars,
- Alpha5 A5Dir/router 2024-train to 2025-score chain.

Remaining caution:

- Any older experiment run, DSAC checkpoint, Alpha6/Alpha7 policy artifact, or cached unified dataset created before this remediation is not automatically clean.
- Promote only artifacts whose input path or manifest references this clean funding run, or retrain/rescore them explicitly.
