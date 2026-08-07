# M7 Red-Team Contract - 2026-05-30

## Status

Active M7 generation and active M7 required-column contracts no longer include the unsupervised GMM / Isolation Forest / VAE models or their derived columns.

Removed active model/meta keys:

- `gmm_volatility`
- `isolation_forest`
- `vae_anomaly`

Removed active generated/required columns:

- `m7_gmm_cluster`
- `m7_gmm_conf`
- `m7_gmm_vol_rank`
- `m7_iso_pred`
- `m7_iso_score`
- `m7_iso_anom`
- `m7_vae_error`
- `m7_vae_threshold`
- `m7_vae_anom`
- `m7_gate_block`
- `m7_size`
- `m7_hdb_label`
- `m7_hdb_prob`

Historical artifacts or CSVs containing those columns are diagnostic-only until retrained/rescored under the current active contract. Do not use any reported backtest that depends on those removed columns as promotion evidence.

## Added Active LightGBM Ensemble

The active M7 contract now includes a supervised LightGBM ensemble:

- artifact: `data/ensemble/supervised/lightgbm_ensemble.json`
- model: `data/ensemble/supervised/lightgbm_ensemble.pkl`
- trainer: `ensemble/supervised/train_lightgbm_ensemble.py`
- training split: 2025 feature/RL frames
- OOS test split: 2026 rebuilt feature/RL frames

The LightGBM ensemble is supervised and uses future path labels only as training targets. Its feature contract blocks `m7_*`, legacy clean-regime prefixes, and old `regime_bull/regime_bear/regime_chop/regime_whipsaw/regime_normal` one-hot inputs.

## Base Model Retrain - 2026-05-30

The following active M7 base artifacts were retrained on 2025 feature/RL frames, evaluated on `data/splits/year_oos/training_features_2026_rebuilt.csv`, and overwritten under the same canonical names only because OOS performance improved:

- `trend_xgb`: 2026 OOS balanced accuracy `0.5248 -> 0.5288`; forbidden feature count `3 -> 0`.
- `multi_target_lgbm`: 2026 OOS direction balanced accuracy `0.5232 -> 0.5240`; quality MAE `0.003874 -> 0.003842`; hold MAE `3.3770 -> 3.3715`.
- `quantile_forest`: 2026 OOS direction accuracy `0.4836 -> 0.4903`; MAE `0.005199 -> 0.005112`; interval width `0.011236 -> 0.011146`; forbidden feature count `1 -> 0`.

`entry_price_model` was retrained as a candidate but not overwritten because 2026 OOS average offset MAE worsened (`0.001676 -> 0.001686`). The runtime M7 wrapper was patched to propagate the existing entry offset outputs instead of dropping them to zero.

Audit outputs:

- Candidate directory: `tmp/causal_regen_20260516/m7_base_clean_feature_retrain_20260530/`
- OOS comparison CSV: `tmp/causal_regen_20260516/m7_base_clean_feature_retrain_20260530/oos_compare.csv`
- Evaluator: `scripts/eval_m7_base_models_oos_20260530.py`

## Direction Contract Update - 2026-05-30

Active downstream M7 feature contracts no longer consume M7 direction heads.

Removed active direction columns:


M7 direction context is removed from downstream AI/M7 feature contracts by user decision. No-trade and risk selection must be handled by `m7_tradeability_score`, `m7_quality_pred`, quantile uncertainty, adverse-probability features, or downstream risk/router layers. Active/live and candidate paths must not require, fabricate, alias, or silently map M7 direction probability columns.

Binary retrain / OOS checks:

- `trend_xgb`: trained on 2025 directional-only labels, excluding triple-barrier `FLAT`; 2026 rebuilt OOS binary bacc `0.7810`, weighted F1 `0.7812`; forbidden feature count `0`.
- `multi_target_lgbm`: direction head trained as binary `DOWN`/`UP`; quality and hold heads remain unchanged; 2026 rebuilt OOS direction bacc `0.7865`; forbidden feature count `0`.

## Allowed Columns

These columns may be used as M7 score/context inputs if the consuming artifact records the exact scored CSV provenance:

- `m7_q10`
- `m7_q90`
- `m7_qwidth`
- `m7_quality_pred`
- `m7_hold_pred`
- `m7_tradeability_score`
- `m7_long_mae_q90`
- `m7_short_mae_q90`
- `m7_long_adverse_prob`
- `m7_short_adverse_prob`

## Conditional Columns

Use only as weak meta/context with explicit provenance. Do not treat these as clean execution targets without recomputation.

- `m7_tail_risk`
- `m7_tp_offset`
- `m7_tp_price`
- `m7_entry_long_price`
- `m7_entry_short_price`
- `m7_target_hold`
- `m7_target_quality`

If entry prices are used, recompute entry offsets from the source prices and current reference price. Do not reuse scored entry-offset columns.

## Removed / Historical-Only Columns

These columns are absent from the active M7 generated/required contract. They must not be required, fabricated, aliased, or silently backfilled in active/live inputs, candidate training inputs, promotion evidence, or backtest claims:

- `m7_gmm_cluster`
- `m7_gmm_conf`
- `m7_gmm_vol_rank`
- `m7_iso_pred`
- `m7_iso_score`
- `m7_iso_anom`
- `m7_vae_error`
- `m7_vae_threshold`
- `m7_vae_anom`
- `m7_gate_block`
- `m7_size`
- `m7_hdb_label`
- `m7_hdb_prob`
- `m7_trend_xgb_dn`
- `m7_trend_xgb_up`
- `m7_mtl_dn`
- `m7_mtl_up`
- `m7_quant_dn`
- `m7_quant_up`
- `m7_prob_dn`
- `m7_prob_up`
- `m7_action`
- `m7_confidence`
- `m7_composite_score`
- `m7_long_edge`
- `m7_short_edge`
- `m7_path_best_side`
- `m7_q50`
- `m7_expected_ret`

## Blockers

- Historical artifacts that still contain removed unsupervised columns must be retrained/rescored before they can become candidate or promotion evidence.
- Silent fallbacks, fabricated compatibility columns, or implicit renames violate the fail-fast feature contract and must not be documented as acceptable behavior.

## Metadata Contract

Any M7-derived artifact or candidate report must record:

- training data path and content hash,
- clean funding run ID,
- scaler contract,
- threshold contract,
- scored CSV path and content hash.

Missing metadata is a blocker, not a documentation gap.

## Remediation Gate

For historical artifacts that used the removed columns:

- Retrain/rescore under the current active M7 contract.
- Retrain and re-evaluate every DSAC, Alpha7, and Alpha8 candidate that consumes M7 columns.
- Promotion evidence must reference the rescored CSV hash and must not rely on removed historical-only columns.
