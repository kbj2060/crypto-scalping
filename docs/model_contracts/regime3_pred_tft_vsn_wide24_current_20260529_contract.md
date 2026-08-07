# Regime3 PRED TFT/VSN With Wide24 Current Contract - 2026-05-29

## Status

Main Regime3 PRED research candidate: `Docs48 all raw`.

Not wired into live trading.

## Purpose

Retest the previous TFT/VSN selected PRED architecture using the new Regime3 input contract:

- Raw causal features
- `regime3_current_wide24_*`

This is the clean replacement experiment for the old Regime4 TFT/VSN PRED stack. It removes whipsaw as a class and predicts only `bull`, `bear`, and `chop`.

## Artifacts

Primary research artifacts:

- VSN36 report: `data/ensemble/reports/regime3_pred_tft_vsn_wide24_current_cleanfunding_20260529_report.json`
- VSN36 artifact dir: `data/ensemble/supervised/regime3_pred_tft_vsn_wide24_current_cleanfunding_20260529/`
- Top74 report: `data/ensemble/reports/regime3_pred_tft_vsn_top74_wide24_current_cleanfunding_20260529_report.json`
- Top74 artifact dir: `data/ensemble/supervised/regime3_pred_tft_vsn_top74_wide24_current_cleanfunding_20260529/`
- Main Docs48 report: `data/ensemble/reports/regime3_pred_tft_vsn_docs48all_wide24_current_cleanfunding_20260529_report.json`
- Main Docs48 artifact dir: `data/ensemble/supervised/regime3_pred_tft_vsn_docs48all_wide24_current_cleanfunding_20260529/`
- Main selection manifest: `data/ensemble/supervised/regime3_pred_tft_vsn_docs48all_wide24_current_cleanfunding_20260529/SELECTED_MAIN_REGIME3_PRED_20260529.json`
- Script: `scripts/train_regime3_pred_tft_vsn_wide24_current_20260529.py`

## Input Contract

Allowed current-regime input prefix:

- `regime3_current_wide24_*`

Forbidden prefixes:

- `clean_regime_2024_unsup_v4_*`
- `clean_regime4_2024_unsup_v1_*`
- `clean_regime4_state24_sticky090_v2_*`
- `regime4_pred_*`

The script fails fast if any non-`regime3_current_wide24_*` regime feature enters the TFT/VSN input set.

## Label Contract

Prediction target:

- `argmax(regime3_current_wide24_{bull,bear,chop}_prob at t + 12 bars)`

This follows the previous PRED philosophy: predict future current-regime state rather than using future path labels directly as input features.

## Results

| candidate | selected features | current selected | 2024Q4 bacc | 2024Q4 logloss | 2026 bacc | 2026 recall bull/bear/chop |
|---|---:|---:|---:|---:|---:|---|
| VSN36 | 36 | 3 | 0.6554 | 0.8139 | 0.6805 | 0.5312 / 0.7539 / 0.7563 |
| Top74 | 74 | 8 | 0.6642 | 0.7109 | 0.6712 | 0.5967 / 0.7424 / 0.6746 |
| Docs40 stable | 39 | 9 | 0.6765 | 0.7408 | 0.6876 | 0.5631 / 0.7697 / 0.7300 |
| Docs48 all | 48 | 9 | 0.6746 | 0.7497 | 0.6911 | 0.6277 / 0.7013 / 0.7442 |
| Docs60 rolled | 60 | 9 | 0.6677 | 0.7623 | 0.6859 | 0.5987 / 0.7834 / 0.6756 |

## Feature-Pack Update

After reviewing Docs Manager feature audit memory, two compact feature packs were tested:

- `docs_regime_pred`: stable audit-approved direction/regime context features only.
- `docs_regime_pred_all`: `docs_regime_pred` plus clean-funding and volume risk-context columns.

The stable pack excludes:

- `close_btc` raw level
- `garch_vol_z`
- raw M7 price outputs
- confirmed-bug regime prefixes

`Docs60 rolled` replaces raw funding/volume scale features with rolling transformations:

- signed `log1p`
- 288-bar rolling IQR z-score
- 288-bar rolling percentile rank
- 12-bar log delta

This removes direct scale exposure, but the initial result did not beat `Docs40 stable` on validation or 2026 balanced accuracy.

Key stable-pack columns:

- `regime3_current_wide24_*`
- `compression_score`
- `atr_pct_rank_288`
- `bb_width_pct_rank_288`
- `btc_volume_impulse_z`
- `vwap_dist_24`
- `vwap_dist_96`
- `cvd_12`
- `cvd_288`
- `eth_btc_ret_spread_12`
- `eth_btc_ret_spread_48`
- `btc_lead_eth_follow_gap_3`
- `price_cvd_divergence`
- `crowding_pressure`
- `long_squeeze_risk`
- `mean_reversion_z`
- `dual_momentum`
- `mtf_trend_1h`
- `mtf_trend_4h`

## Interpretation

Both TFT/VSN variants materially outperform the Mamba PRED direction experiment. The result supports using the old TFT/VSN selected family for directional Regime3 PRED.

User-selected main candidate: `Docs48 all raw`.

Reason: `Docs48 all` has the best 2026 balanced accuracy and bull recall among the tested clean Regime3 PRED variants.

Risk note: `Docs40 stable` remains the more validation-defensive candidate. `Docs48 all` adds clean-funding and raw volume risk-context columns, so it should be treated as an aggressive main research candidate until downstream Alpha integration or walk-forward confirms stability.

`Docs60 rolled` is cleaner than `Docs48 all` from a distribution-scale standpoint, but current metrics do not justify selecting it over `Docs40 stable`.
