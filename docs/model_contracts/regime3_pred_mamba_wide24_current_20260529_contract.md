# Regime3 PRED Mamba With Wide24 Current Contract - 2026-05-29

## Status

Research artifact. Not wired into live trading.

## Purpose

Retrain the Regime3 future/risk predictor after confirming `wide24` as the next Regime3 CURRENT candidate.

The intent is to restore the useful behavior of the old PRED stack, which relied heavily on current-regime context, without using legacy Regime4 inputs.

## Selected Candidate

Selected research candidate:

- `regime3_pred_mamba_wide24_current_cleanfunding_20260529`
- Artifact directory: `data/ensemble/supervised/regime3_pred_mamba_wide24_current_cleanfunding_20260529/`
- Report: `data/ensemble/reports/regime3_pred_mamba_wide24_current_cleanfunding_20260529_report.json`
- Script: `scripts/train_regime3_pred_mamba_wide24_current_20260529.py`

## Input Contract

Allowed current-regime input prefix:

- `regime3_current_wide24_*`

Selected model uses six current features:

- `regime3_current_wide24_bull_prob`
- `regime3_current_wide24_bear_prob`
- `regime3_current_wide24_chop_prob`
- `regime3_current_wide24_confidence`
- `regime3_current_wide24_entropy`
- `regime3_current_wide24_margin`

Forbidden prefixes:

- `clean_regime_2024_unsup_v4_*`
- `clean_regime4_2024_unsup_v1_*`
- `clean_regime4_state24_sticky090_v2_*`
- `regime4_pred_*`

## Rejected Variant

Also tested `regime3_pred_mamba_wide24_current_plus_cleanfunding_20260529`, which added:

- `regime3_current_wide24_directional_bias`
- `regime3_current_wide24_trend_prob`
- `regime3_current_wide24_range_prob`

This improved validation loss but degraded 2025/2026 balanced accuracy and risk AUC. Treat it as rejected for now.

## 2026 Forward Comparison

| model | h12 bacc | h24 bacc | h48 bacc | instability AUC | false-breakout AUC |
|---|---:|---:|---:|---:|---:|
| raw no-current | 0.3654 | 0.3836 | 0.3346 | 0.8147 | 0.7043 |
| wide24 current 6 | 0.3635 | 0.3624 | 0.3254 | 0.8840 | 0.7217 |
| wide24 current 9 | 0.3522 | 0.3216 | 0.3097 | 0.8297 | 0.7204 |

## Interpretation

Adding confirmed CURRENT context improves some headline accuracy and materially improves risk prediction, especially `instability_prob` and `false_breakout_risk`.

It does not yet solve directional future regime recall. Use this artifact as risk/context input, not as an action-direction owner.
