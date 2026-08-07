# Alpha4.3 HGB ATR TP/SL Action Score Test - 2026-05-17

## Purpose

Evaluate the original HGB sparse hold-aware `tp_sl_action_score` layer after replacing the fixed TP/SL label barriers with ATR-scaled barriers.

This layer remains a parent input feature only. It is not an exit gate and does not choose live TP/SL orders.

## Feature Contract

Feature: `tp_sl_action_score`

Model:

- Long model: `SimpleImputer + HistGradientBoostingRegressor`
- Short model: `SimpleImputer + HistGradientBoostingRegressor`
- `max_iter=180`
- `learning_rate=0.04`
- `l2_regularization=0.12`
- `min_samples_leaf=35`

Score rule:

```text
if max(long_edge, short_edge) <= 0:
    tp_sl_action_score = 0
elif long_edge >= short_edge:
    tp_sl_action_score = +long_edge
else:
    tp_sl_action_score = -short_edge
```

## Label Contract

- Entry reference: next bar open.
- Horizon: 48 bars.
- ATR source: causal OHLC true range rolling mean, window 14.
- TP barrier: `ATR_pct * 3.0`.
- SL barrier: `ATR_pct * 1.5`.
- Same-bar TP/SL tie: SL wins.
- Candidate TP/SL/max-hold values are not used.
- 2025 generation: monthly walk-forward OOF with 50-row purge gap.
- 2026 generation: final HGB pair trained on all 2025 labels only.

Observed 2025 ATR barrier stats:

- ATR pct mean: 0.2910%
- ATR pct median: 0.2487%
- TP mean: 0.8730%
- TP median: 0.7461%
- SL mean: 0.4365%
- SL median: 0.3730%

## Artifacts

- Feature audit: `/home/llewyn/crypto-scalping/tmp/causal_regen_20260516/alpha4_3_hgb_atr3_tp_sl_action_score_20260517/tp_sl_path_edge_feature_audit.json`
- Train CSV: `/home/llewyn/crypto-scalping/tmp/causal_regen_20260516/alpha4_3_hgb_atr3_tp_sl_action_score_20260517/trade_candidates_2025_patchtst__tide__dlinear.csv`
- Eval CSV: `/home/llewyn/crypto-scalping/tmp/causal_regen_20260516/alpha4_3_hgb_atr3_tp_sl_action_score_20260517/trade_candidates_2026_patchtst__tide__dlinear.csv`
- Parent train report: `/home/llewyn/crypto-scalping/tmp/causal_regen_20260516/alpha4_3_hgb_atr3_tp_sl_action_score_20260517/alpha4_3_hgb_atr3_parent_train_summary.json`
- Teacher ablation report: `/home/llewyn/crypto-scalping/tmp/causal_regen_20260516/alpha4_3_hgb_atr3_tp_sl_action_score_20260517/alpha4_3_hgb_atr3_teacher_ablation_summary.json`

## Feature Distribution

2025 train:

- mean: -0.01012
- std: 0.06912
- zero rate: 89.73%
- positive rate: 2.53%
- negative rate: 7.74%

2026 eval:

- mean: +0.00215
- std: 0.02882
- zero rate: 95.06%
- positive rate: 3.57%
- negative rate: 1.36%

The ATR barrier made the feature much sparser than the fixed-barrier HGB score.

## Backtest Results

Full downstream HGB retrain, with teacher/deep path still available:

- cost1 PnL: +55.50%
- cost1 MDD: -28.80%
- cost2 PnL: +50.46%
- cost3 PnL: +44.86%
- trades: 61

No-teacher/no-deep ablation:

| Variant | Cost1 PnL | MDD | Cost2 PnL | Cost3 PnL | Trades |
| --- | ---: | ---: | ---: | ---: | ---: |
| `parent_direct_raw_no_teacher` | +96.94% | -24.34% | +87.88% | +84.94% | 69 |
| `parent_direct_scaled_no_teacher` | +72.34% | -19.46% | +68.05% | +73.96% | 66 |
| `teacher_constrained` | +88.80% | -31.45% | +76.74% | +103.56% | 66 |

Validation selected `parent_direct_raw_no_teacher`; verdict was `remove_teacher`.

## Baseline Comparison

Existing fixed-barrier HGB Alpha4.3 reference:

- cost1 PnL: +183.42%
- MDD: -21.99%
- cost2 PnL: +169.76%
- cost3 PnL: +79.27%
- trades: 66

## Verdict

Do not promote the ATR×3.0/ATR×1.5 HGB TP/SL score.

It is better than the LightGBM Quantile score tested earlier, especially under cost3, but it still materially underperforms the existing fixed-barrier HGB Alpha4.3 reference. The dynamic ATR labels are too tight on this 5-minute frame: median TP is only about 0.75% and median SL about 0.37%, which makes both long and short targets strongly negative and causes the sparse HGB score to fire on too few rows.

Next useful sweep would be wider ATR barriers, for example TP/SL of `ATR×6/ATR×3`, `ATR×8/ATR×4`, or a floor such as `max(fixed_pct, ATR_mult * ATR_pct)`.
