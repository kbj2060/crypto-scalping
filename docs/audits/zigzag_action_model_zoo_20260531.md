# ZigZag Action Model Zoo Audit - 2026-05-31

## Scope

Retrained direct action/direction label model families against the active ZigZag 3-class label contract:

- `zigzag_action`: `0=CASH`, `1=LONG`, `2=SHORT`
- Label source: `tmp/causal_regen_20260516/zigzag_action_labels_20260531`
- Script: `scripts/train_zigzag_action_model_zoo_20260531.py`
- Artifact root: `tmp/causal_regen_20260516/zigzag_action_model_zoo_20260531`

This is a comparison/promotion audit. It does not automatically change the Omega1 active feature list.

## Contract Guards

- The trainer fails if `wave3_action` appears in active label files.
- No alias/fallback label mapping is used.
- Inputs exclude `teacher_*`, `m7_*`, `a5dir_*`, `ai_*`, `pred_*`, `conf_*`, Regime4, `regime3_pred_*`, labels, targets, future, PnL, `zigzag_*`, and `wave3_*`.
- Score splits: `2024 train -> 2025 score`, and `2025 train -> 2026 score`.

## Tested Families

- `m7_trend_xgb`: original `ensemble/supervised/train_trend_xgb.py`, tested as `trend_xgb_like_lgbm` and `trend_xgb_like_xgb`.
- `m7_multitarget_lgbm`: original `ensemble/supervised/train_multitarget_lgbm.py`, tested as `multitarget_lgbm_like`.
- `m7_quantile_forest`: original `ensemble/supervised/train_quantile_forest.py`, tested as `quantile_feature_like_lgbm`. This is an action-proxy because the original model is a return-quantile regressor, not a direct action classifier.
- `alpha5_hgb_action_master`: original `scripts/tune_alpha5_9_hgb_action_master_20260518.py`, tested as `alpha_hgb_action_master_like`.
- `alpha5/alpha6 LGBM/CatBoost action masters`: tested as `alpha_lgbm_action_master_like` and `alpha_catboost_action_master_like`.

## 2025 Score Metrics

| model | train bacc | 2025 bacc | 2025 OVR AUC | pred cash | pred long | pred short |
|---|---:|---:|---:|---:|---:|---:|
| `alpha_catboost_action_master_like` | 0.732816 | 0.568380 | 0.762591 | 34750 | 30044 | 40307 |
| `trend_xgb_like_xgb` | 0.712349 | 0.553729 | 0.751863 | 32885 | 26948 | 45268 |
| `multitarget_lgbm_like` | 0.936124 | 0.541560 | 0.742733 | 22403 | 36244 | 46454 |
| `trend_xgb_like_lgbm` | 0.933933 | 0.540559 | 0.739884 | 22573 | 37159 | 45369 |
| `quantile_feature_like_lgbm` | 0.940273 | 0.539956 | 0.740051 | 21825 | 39853 | 43423 |
| `alpha_hgb_action_master_like` | 0.912840 | 0.534123 | 0.738855 | 21039 | 39610 | 44452 |
| `alpha_lgbm_action_master_like` | 0.978460 | 0.529043 | 0.744873 | 15387 | 44111 | 45603 |

## 2026 OOS Metrics

| model | train bacc | 2026 bacc | 2026 OVR AUC | pred cash | pred long | pred short |
|---|---:|---:|---:|---:|---:|---:|
| `alpha_catboost_action_master_like` | 0.717121 | 0.565474 | 0.755714 | 4298 | 6012 | 6587 |
| `trend_xgb_like_xgb` | 0.709720 | 0.555528 | 0.750837 | 3747 | 6688 | 6462 |
| `quantile_feature_like_lgbm` | 0.939796 | 0.536903 | 0.744360 | 2758 | 7616 | 6523 |
| `alpha_hgb_action_master_like` | 0.897735 | 0.535653 | 0.739463 | 3021 | 6485 | 7391 |
| `trend_xgb_like_lgbm` | 0.935995 | 0.534785 | 0.734313 | 2687 | 7633 | 6577 |
| `multitarget_lgbm_like` | 0.936802 | 0.528298 | 0.735595 | 2679 | 7686 | 6532 |
| `alpha_lgbm_action_master_like` | 0.971084 | 0.515107 | 0.738806 | 2051 | 6967 | 7879 |

## Readout

- Best practical ZigZag direct-action model in this pass: `alpha_catboost_action_master_like`, 2026 BAcc `0.565474`, OVR AUC `0.755714`.
- Best direct Trend-XGB-style retrain: `trend_xgb_like_xgb`, 2026 BAcc `0.555528`, OVR AUC `0.750837`.
- LGBM variants show useful OVR AUC but strong train/score gaps, so they need stronger regularization or time-series CV before active/live promotion.
- `quantile_feature_like_lgbm` should be treated as a proxy test, not an exact replacement for the original quantile-return model.

## Artifacts

- Summary: `tmp/causal_regen_20260516/zigzag_action_model_zoo_20260531/zigzag_action_model_zoo_summary.json`
- Flat metrics: `tmp/causal_regen_20260516/zigzag_action_model_zoo_20260531/zigzag_action_model_zoo_flat_metrics.csv`
- Per-model score/model artifacts are under `tmp/causal_regen_20260516/zigzag_action_model_zoo_20260531/<model>/`.
