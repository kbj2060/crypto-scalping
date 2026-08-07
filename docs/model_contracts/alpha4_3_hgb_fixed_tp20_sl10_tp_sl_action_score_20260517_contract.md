# Alpha4.3 HGB Fixed TP2.0 SL1.0 Action Score Test - 2026-05-17

## Purpose

Evaluate the original HGB sparse hold-aware `tp_sl_action_score` layer with fixed TP/SL labels changed from `1.8% / 1.0%` to `2.0% / 1.0%`.

This feature remains a parent input only. It does not set live TP/SL orders.

## Feature Contract

Model:

- Long model: `SimpleImputer + HistGradientBoostingRegressor`
- Short model: `SimpleImputer + HistGradientBoostingRegressor`
- `max_iter=180`
- `learning_rate=0.04`
- `l2_regularization=0.12`
- `min_samples_leaf=35`

Label:

- Entry reference: next bar open.
- Horizon: 48 bars.
- Fixed TP: 2.0%.
- Fixed SL: 1.0%.
- Same-bar TP/SL tie: SL wins.
- Candidate TP/SL/max-hold values are not used.
- 2025 generation: monthly walk-forward OOF with 50-row purge gap.
- 2026 generation: final HGB pair trained on all 2025 labels only.

Score:

```text
if max(long_edge, short_edge) <= 0:
    tp_sl_action_score = 0
elif long_edge >= short_edge:
    tp_sl_action_score = +long_edge
else:
    tp_sl_action_score = -short_edge
```

## Artifacts

- Feature audit: `/home/llewyn/crypto-scalping/tmp/causal_regen_20260516/alpha4_3_hgb_fixed_tp20_sl10_tp_sl_action_score_20260517/tp_sl_path_edge_feature_audit.json`
- Train CSV: `/home/llewyn/crypto-scalping/tmp/causal_regen_20260516/alpha4_3_hgb_fixed_tp20_sl10_tp_sl_action_score_20260517/trade_candidates_2025_patchtst__tide__dlinear.csv`
- Eval CSV: `/home/llewyn/crypto-scalping/tmp/causal_regen_20260516/alpha4_3_hgb_fixed_tp20_sl10_tp_sl_action_score_20260517/trade_candidates_2026_patchtst__tide__dlinear.csv`
- Parent train report: `/home/llewyn/crypto-scalping/tmp/causal_regen_20260516/alpha4_3_hgb_fixed_tp20_sl10_tp_sl_action_score_20260517/alpha4_3_hgb_fixed_tp20_sl10_parent_train_summary.json`
- Teacher ablation report: `/home/llewyn/crypto-scalping/tmp/causal_regen_20260516/alpha4_3_hgb_fixed_tp20_sl10_tp_sl_action_score_20260517/alpha4_3_hgb_fixed_tp20_sl10_teacher_ablation_summary.json`

## Feature Distribution

2025 train:

- mean: -0.01959
- std: 0.08488
- zero rate: 75.71%
- positive rate: 5.07%
- negative rate: 19.22%

2026 eval:

- mean: -0.00100
- std: 0.05469
- zero rate: 82.95%
- positive rate: 10.52%
- negative rate: 6.53%

## Backtest Results

Full downstream HGB retrain:

- cost1 PnL: +58.11%
- cost1 MDD: -26.57%
- cost2 PnL: +23.45%
- cost3 PnL: +23.27%
- trades: 81

No-teacher/no-deep ablation:

| Variant | Cost1 PnL | MDD | Cost2 PnL | Cost3 PnL | Trades |
| --- | ---: | ---: | ---: | ---: | ---: |
| `parent_direct_raw_no_teacher` | +83.07% | -26.98% | +79.07% | +103.12% | 70 |
| `parent_direct_scaled_no_teacher` | +83.07% | -26.98% | +79.07% | +103.12% | 70 |
| `teacher_constrained` | +49.14% | -31.58% | +45.50% | +50.33% | 56 |

Validation selected `parent_direct_raw_no_teacher`; verdict was `remove_teacher`.

## Baseline Comparison

Existing fixed `1.8% / 1.0%` HGB Alpha4.3 reference:

- `parent_direct_scaled_no_teacher`: cost1 +183.42%, MDD -21.99%, cost2 +169.76%, cost3 +79.27%, trades 66.
- `parent_direct_raw_no_teacher`: cost1 +169.37%, MDD -22.74%, cost2 +107.66%, cost3 +100.75%, trades 73.

ATR `3.0 / 1.5` HGB comparison:

- `parent_direct_raw_no_teacher`: cost1 +96.94%, MDD -24.34%, cost2 +87.88%, cost3 +84.94%, trades 69.

## Verdict

Do not promote fixed `2.0% / 1.0%`.

The wider TP increases cost3 for the raw no-teacher variant versus the existing selected Alpha4.3, but it cuts cost1 and cost2 too much and worsens MDD. The existing fixed `1.8% / 1.0%` score remains the stronger Alpha4.3 reference.
