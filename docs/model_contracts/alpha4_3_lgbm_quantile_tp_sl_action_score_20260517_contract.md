# Alpha4.3 TP/SL Direction Feature - LightGBM Quantile Regression

Date: 2026-05-17

## Purpose

Test a TP/SL-based single directional parent input feature using LightGBM Quantile Regression.

The feature is not an exit gate and not a candidate-specific TP/SL/max-hold predictor. It is a single signed parent feature:

- `tp_sl_action_score > 0`: long path edge is better than short path edge.
- `tp_sl_action_score < 0`: short path edge is better than long path edge.
- `tp_sl_action_score = 0`: insufficient walk-forward history or optional deadband.

## Label Contract

- Entry price: next bar open.
- Horizon: 48 bars.
- Fixed TP: 1.8%.
- Fixed SL: 1.0%.
- Same-bar TP/SL tie: SL wins, conservatively.
- Long and short labels are computed separately.
- Label return is clipped to `[-0.5, 0.5]` after scaling by the fixed TP/SL path convention.

## Model

Script: `/home/llewyn/crypto-scalping/scripts/build_alpha4_tp_sl_path_edge_feature_20260517.py`

Model family:

- Long side: LightGBM quantile regressors at q25, q50, q75.
- Short side: LightGBM quantile regressors at q25, q50, q75.
- Estimators in this validation run: 50.

Side edge:

```text
side_edge = q50 - 0.50 * max(0, q50 - q25) - 0.10 * max(0, q75 - q25)
```

Final feature:

```text
tp_sl_action_score = long_side_edge - short_side_edge
```

This relative signed edge is important. The first attempted absolute-positive policy collapsed all rows to zero because both long and short absolute path edges were often negative under the conservative fixed TP/SL label.

## Leak Prevention

2025 training feature generation uses monthly walk-forward OOF.

- Prediction month is never used for fitting that month.
- A purge gap of `horizon + 2 = 50` rows is applied before each prediction month.
- 2026 evaluation feature is predicted by models fitted on 2025 labels only.
- `selection_uses_2026 = false`.

Audit:

`/home/llewyn/crypto-scalping/tmp/causal_regen_20260516/alpha4_3_purged_lgbm_quantile_signed_tp_sl_action_score_20260517/tp_sl_path_edge_feature_audit.json`

## Feature Distribution

Generated CSVs:

- Train: `/home/llewyn/crypto-scalping/tmp/causal_regen_20260516/alpha4_3_purged_lgbm_quantile_signed_tp_sl_action_score_20260517/trade_candidates_2025_patchtst__tide__dlinear.csv`
- Eval: `/home/llewyn/crypto-scalping/tmp/causal_regen_20260516/alpha4_3_purged_lgbm_quantile_signed_tp_sl_action_score_20260517/trade_candidates_2026_patchtst__tide__dlinear.csv`

2025 train:

- mean: -0.03916
- std: 0.11509
- zero rate: 25.01%
- positive rate: 25.81%
- negative rate: 49.19%

2026 eval:

- mean: +0.01572
- std: 0.09033
- zero rate: 3.81%
- positive rate: 55.00%
- negative rate: 41.19%

## Backtest Results

All backtests used the corrected Alpha3 limit-close execution contract and the regenerated causal feature frame.

### LightGBM Parent + LightGBM Quantile TP/SL Score

Report:

`/home/llewyn/crypto-scalping/tmp/causal_regen_20260516/alpha4_3_purged_lgbm_quantile_signed_tp_sl_action_score_20260517/alpha4_3_lgbm_quantile_parent_train_summary.json`

Full downstream OOS:

- cost1 PnL: +49.26%
- cost1 MDD: -33.03%
- cost2 PnL: +57.30%
- cost3 PnL: +61.07%
- trades: 78

No-teacher/no-deep ablation:

Report:

`/home/llewyn/crypto-scalping/tmp/causal_regen_20260516/alpha4_3_purged_lgbm_quantile_signed_tp_sl_action_score_20260517/alpha4_3_lgbm_quantile_teacher_ablation_summary.json`

- `parent_direct_raw_no_teacher`: cost1 +59.39%, MDD -29.66%, cost2 +56.33%, cost3 +53.52%, trades 37.
- `parent_direct_scaled_no_teacher`: cost1 +45.10%, MDD -35.41%, cost2 +44.46%, cost3 +42.15%, trades 37.
- `teacher_constrained`: cost1 +60.25%, MDD -31.41%, cost2 +58.53%, cost3 +53.47%, trades 34.

### HGB Parent + LightGBM Quantile TP/SL Score

Report:

`/home/llewyn/crypto-scalping/tmp/causal_regen_20260516/alpha4_3_purged_lgbm_quantile_signed_tp_sl_action_score_20260517/alpha4_3_hgb_parent_lgbm_quantile_score_train_summary.json`

Full downstream OOS:

- cost1 PnL: +4.95%
- cost1 MDD: -36.47%
- cost2 PnL: +2.30%
- cost3 PnL: +3.76%
- trades: 60

No-teacher/no-deep ablation:

Report:

`/home/llewyn/crypto-scalping/tmp/causal_regen_20260516/alpha4_3_purged_lgbm_quantile_signed_tp_sl_action_score_20260517/alpha4_3_hgb_lgbm_quantile_teacher_ablation_summary.json`

- `parent_direct_raw_no_teacher`: cost1 +58.61%, MDD -23.25%, cost2 +55.30%, cost3 +49.98%, trades 74.
- `parent_direct_scaled_no_teacher`: cost1 +61.61%, MDD -25.35%, cost2 +36.04%, cost3 +30.41%, trades 63.
- `teacher_constrained`: cost1 +6.84%, MDD -33.52%, cost2 +0.97%, cost3 +5.96%, trades 54.

## Baseline Comparison

Current Alpha4.3 no-teacher/no-deep reference:

Report:

`/home/llewyn/crypto-scalping/tmp/causal_regen_20260516/alpha4_3_no_teacher_no_deep_20260517/alpha4_3_no_teacher_no_deep_summary.json`

- cost1 PnL: +183.42%
- cost1 MDD: -21.99%
- cost2 PnL: +169.76%
- cost3 PnL: +79.27%
- trades: 66

## Verdict

Do not promote this LightGBM Quantile TP/SL score into the live Alpha4.3 stack.

The feature generation is now leak-safe and technically valid, but the retrained stacks underperform the existing Alpha4.3 reference by a wide margin. The most likely reason is that the fixed TP/SL path label is too conservative and over-penalizes candidates that later become profitable under the existing runner/execution lifecycle. It also compresses the parent toward fewer, stop-loss-heavy trades.

Keep the artifact as a research feature only. For the next iteration, use the same purged walk-forward framework but change the target from fixed TP/SL first-touch return to realized lifecycle utility under the corrected Alpha3 execution ledger.
