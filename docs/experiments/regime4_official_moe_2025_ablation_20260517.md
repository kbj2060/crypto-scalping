# Regime4 Official MoE 2025 Ablation

Date: 2026-05-17

## Purpose

Test whether the official 4-class Regime4 sidecars improve a downstream trading MoE.

Official taxonomy:

```text
bull
bear
chop
whipsaw
```

No `normal`, `risk_off`, or `transition` class is used.

## Inputs

Current regime sidecar:

```text
/home/llewyn/crypto-scalping/data/ensemble/supervised/clean_regime4_raw_state12_v1_20260517/training_features_2025_clean_regime4_raw_state12_v1.csv
```

Future regime sidecar:

```text
/home/llewyn/crypto-scalping/data/ensemble/supervised/regime4_pred_tft_vsn_h12_official_20260517/training_features_2025_regime4_pred_tft_vsn_selected.csv
```

Evaluation script:

```text
/home/llewyn/crypto-scalping/scripts/eval_regime4_official_moe_2025_20260517.py
```

Report:

```text
/home/llewyn/crypto-scalping/data/ensemble/reports/regime4_official_moe_2025_ablation_20260517.json
```

## Design

The experiment compares four variants:

```text
baseline         raw + AI/M7/market features, no Regime4 sidecars
regime4_current current HMM Regime4 features, current Regime4 soft gate
regime4_future  future TFT Regime4 features, future Regime4 soft gate
regime4_both    current + future Regime4 features, future Regime4 soft gate
```

Each MoE variant trains:

```text
global direction classifier
4 regime-specific direction experts when a gate is available
global long/short adverse-risk regressors
selection runtime grid using 2025 selection split
holdout replay using the selected config
```

This is a vectorized diagnostic, not a live/runtime-native promotion test.

## Split

```text
fit       < 2025-09-01
selection 2025-09-01 .. < 2025-11-01
holdout   >= 2025-11-01
```

Rows:

```text
merged  105064
labeled 105027
```

Sidecar audit:

```text
clean_regime4 prob sum min/max 0.9999999999999996 / 1.0000000000000002, NaN 0
regime4_pred  prob sum min/max 0.9999999999999994 / 1.0000000000000002, NaN 0
```

## Holdout Result

Cost1 ranking:

| Variant | PnL | MDD | Trade Sharpe | Trades | Direction Accuracy | Experts |
|---|---:|---:|---:|---:|---:|---:|
| regime4_both | -13.39% | -14.08% | -1.99 | 265 | 0.4339 | 4 |
| baseline | -13.44% | -15.31% | -1.97 | 270 | 0.4331 | 0 |
| regime4_future | -17.30% | -18.57% | -2.72 | 268 | 0.4343 | 4 |
| regime4_current | -17.88% | -18.18% | -3.13 | 258 | 0.4294 | 4 |

Cost stress for the best variant, `regime4_both`:

```text
cost1 PnL -13.39%, MDD -14.08%, trade_sharpe -1.99
cost2 PnL -26.36%, MDD -26.54%, trade_sharpe -4.28
cost3 PnL -37.39%, MDD -37.47%, trade_sharpe -6.56
```

## Verdict

`regime4_both` slightly improves MDD and PnL versus baseline in this diagnostic, but the absolute holdout PnL is negative and cost stress deteriorates sharply.

Verdict:

```text
reject for promotion
keep Regime4 official sidecars as candidate context/gating features
do not wire this MoE layer into live or Alpha3 promotion path from this result
```

## Next Actions

Do not tune this exact vectorized MoE further as a promotion path.

Useful follow-up tests:

```text
1. Use Regime4 as a veto/size modulation feature on an already profitable Alpha3/DSAC parent instead of training a standalone direction owner.
2. Generate 2026 Regime4 sidecars and run a fixed-OOS diagnostic before any contract promotion.
3. If MoE is revisited, train experts on trade expectancy/owner selection, not only long/short/no-trade direction labels.
```
