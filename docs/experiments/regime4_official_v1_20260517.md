# Regime4 Official V1 Experiment

Date: 2026-05-17

## Summary

The project now has an official 4-class regime experiment line:

```text
bull
bear
chop
whipsaw
```

`normal` is removed from the active MoE taxonomy because it collapsed in the 5-class HMM and did not provide a learnable future target.

## Current Regime Result

4-class HMM raw-state12:

```text
accuracy          0.6436
balanced_accuracy 0.6203
log_loss          0.9516
```

2025 hard argmax distribution:

```text
bull     32577
bear     32270
chop     25260
whipsaw  14957
```

Official sidecar:

```text
/home/llewyn/crypto-scalping/data/ensemble/supervised/clean_regime4_raw_state12_v1_20260517/training_features_2025_clean_regime4_raw_state12_v1.csv
```

## Future Predictor Selection

Validated options:

```text
4-class h36 all features
accuracy          0.4161
balanced_accuracy 0.4190
log_loss          1.2760

4-class h36 VSN selected
accuracy          0.3968
balanced_accuracy 0.4030
log_loss          1.2827

4-class multi-horizon h12 head
accuracy          0.5694
balanced_accuracy 0.5570
log_loss          0.9767

4-class h12 VSN selected official
accuracy          0.6009
balanced_accuracy 0.6004
log_loss          0.9228
```

The official future predictor is `VSN selected + h12`.

Official sidecar:

```text
/home/llewyn/crypto-scalping/data/ensemble/supervised/regime4_pred_tft_vsn_h12_official_20260517/training_features_2025_regime4_pred_tft_vsn_selected.csv
```

Official reproduction command:

```bash
venv/bin/python scripts/build_regime4_pred_tft_vsn_select_20260517.py \
  --horizon 12 \
  --out-dir data/ensemble/supervised/regime4_pred_tft_vsn_h12_official_20260517 \
  --report data/ensemble/reports/regime4_pred_tft_vsn_h12_official_20260517_report.json
```

The validation numbers below are selection metrics, because the VSN-selected feature set and h12 choice were chosen using validation results. They are suitable for freezing the candidate line before downstream testing, not for claiming final OOS performance.

## Official Selected Features

```text
sum_toptrader_long_short_ratio
close_btc
sum_open_interest_value
btc_corr_60
clean_regime4_2024_unsup_v1_bear_prob
pred_mdjd
clean_regime4_2024_unsup_v1_trend_prob
hour_sin
breakout_strength
clean_regime4_2024_unsup_v1_chop_prob
mean_reversion_z
ofi_acceleration
chop_index
clean_regime4_2024_unsup_v1_range_prob
clean_regime4_2024_unsup_v1_micro_prob
clean_regime4_2024_unsup_v1_bull_prob
squeeze_power
bb_width_z
funding_price_divergence
amihud_illiquidity_z
kalman_velocity
taker_acceleration
hour_cos
cvp_cluster_position
mtf_trend_1h
clean_regime4_2024_unsup_v1_entropy
session_europe
clean_regime4_2024_unsup_v1_instability_prob
trade_intensity
whale_retail_ratio
taker_buy_quote
trades
hma_slope
macd_hist
net_taker_ratio
```

## Validation

Official h12 VSN-selected validation:

```text
rows              26484
accuracy          0.6009
balanced_accuracy 0.6004
log_loss          0.9228
```

2025 prediction distribution:

```text
bull     30202
bear     29126
chop     24013
whipsaw  21723
```

2025 sidecar checks:

```text
rows      105064
columns   13
NaN       0
prob sum  1.0
```

## Next Integration Step

Backtests should merge these two sidecars by `timestamp`:

```text
/home/llewyn/crypto-scalping/data/ensemble/supervised/clean_regime4_raw_state12_v1_20260517/training_features_2025_clean_regime4_raw_state12_v1.csv
/home/llewyn/crypto-scalping/data/ensemble/supervised/regime4_pred_tft_vsn_h12_official_20260517/training_features_2025_regime4_pred_tft_vsn_selected.csv
```

Use `regime4_pred_*` as future expert weights and `clean_regime4_2024_unsup_v1_*` as current-state context.

The multi-horizon 12/36/72 experiment remains auxiliary. Do not merge its sidecar as the official future regime feature unless a later backtest explicitly promotes it.

## Downstream MoE Diagnostic

Initial downstream diagnostic:

```text
/home/llewyn/crypto-scalping/scripts/eval_regime4_official_moe_2025_20260517.py
/home/llewyn/crypto-scalping/data/ensemble/reports/regime4_official_moe_2025_ablation_20260517.json
/home/llewyn/crypto-scalping/docs/experiments/regime4_official_moe_2025_ablation_20260517.md
```

2025 holdout result:

```text
best variant   regime4_both
cost1 PnL      -13.39%
cost1 MDD      -14.08%
trade Sharpe   -1.99
trades         265
accuracy       0.4339
```

Verdict: reject for promotion. Keep Regime4 sidecars as candidate context/gating features, but do not treat this standalone direction MoE as a live candidate.

## Fixed Preprocessing Update

The fixed Regime4 + TP/SL preprocessing contract uses a no-mdjd future Regime4 artifact:

```text
/home/llewyn/crypto-scalping/data/ensemble/supervised/regime4_pred_tft_h12_nomdjd_all74_20260517/training_features_2025_regime4_pred_tft_vsn_selected.csv
/home/llewyn/crypto-scalping/data/ensemble/supervised/regime4_pred_tft_h12_nomdjd_all74_20260517/training_features_2026_rebuilt_regime4_pred_tft_vsn_selected.csv
```

`pred_mdjd` and `conf_mdjd` are excluded. This removes the 2026 median-fallback issue and improved validation versus the previous h12 selected artifact:

```text
prior h12 selected accuracy 0.6009, log_loss 0.9228
no-mdjd all74 accuracy      0.6219, log_loss 0.8989
```

Canonical fixed preprocessing files:

```text
/home/llewyn/crypto-scalping/tmp/causal_regen_20260516/fixed_regime4_tp18_sl10_preprocess_20260517/trade_candidates_2025_regime4_tp18_sl10_fixed.csv
/home/llewyn/crypto-scalping/tmp/causal_regen_20260516/fixed_regime4_tp18_sl10_preprocess_20260517/trade_candidates_2026_regime4_tp18_sl10_fixed.csv
```
