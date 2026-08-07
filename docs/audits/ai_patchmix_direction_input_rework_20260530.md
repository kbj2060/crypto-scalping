# AI PatchTSMixer Direction Input Rework - 2026-05-30

## Purpose

Rework the AI direction feature input set using the document-manager feature audit, while preserving the upstream contract:

- no `teacher_*`
- no `m7_*`
- no `a5dir_*`
- no existing `ai_*` / TSFM outputs
- no regime sidecar outputs
- no labels, targets, future path, or PnL-derived columns

## Scripts

- Builder: `scripts/build_ai_patchmix_direction_core_20260530.py`
- Full audit runner: `scripts/run_ai_patchmix_direction_core_audit_v2_20260530.sh`
- Compact audit runner: `scripts/run_ai_patchmix_direction_core_audit_compact_20260530.sh`

## Input Profiles

### v1 baseline

The original 19-feature clean market core:

- returns: `ret_1`, `ret_3`, `ret_6`, `ret_12`, `ret_24`
- volatility/compression: `atr14_pct`, `realized_vol_24`, `compression_ratio`
- clean funding/OI: `last_funding_rate`, `funding_pressure`, `oi_change_rate`
- flow/microstructure: `smart_money_flow`, `ofi_acceleration`, `net_taker_ratio`, `taker_acceleration`, `cvp_volume_imbalance`
- execution shape: `vwap_dev_48`, `lower_wick_ratio`, `upper_wick_ratio`

### audit_full

Adds all direct audited context features that are present in the year OOS splits:

`funding_abs`, `funding_roc_288`, `funding_price_divergence`, `mta_funding`, `long_squeeze_risk`, `crowding_pressure`, `crowded_short_squeeze_risk`, `crowded_long_unwind_risk`, `compression_score`, `atr_pct_rank_288`, `bb_width_pct_rank_288`, `vwap_dist_24`, `vwap_dist_96`, `anchored_vwap_session_dist`, `cvd_288`, `eth_btc_ret_spread_12`, `btc_lead_eth_follow_gap_3`, `btc_volume_impulse_z`, `range_contraction_breakout_dir`, `distance_to_day_high_low_pct`, `price_cvd_divergence`, `funding_oi_divergence`, `hour_sin`, `hour_cos`, `session_us`.

### audit_compact

Keeps the stronger directional/context subset from the document-manager audit:

`funding_roc_288`, `long_squeeze_risk`, `crowding_pressure`, `crowded_short_squeeze_risk`, `crowded_long_unwind_risk`, `compression_score`, `atr_pct_rank_288`, `bb_width_pct_rank_288`, `vwap_dist_96`, `cvd_288`, `eth_btc_ret_spread_12`, `btc_lead_eth_follow_gap_3`, `btc_volume_impulse_z`, `range_contraction_breakout_dir`, `price_cvd_divergence`, `hour_sin`, `hour_cos`, `session_us`.

## Result Summary

Metric is evaluated against the same local triple-barrier labels used by the builder. Higher is better.

| profile/year | h12 bacc | h12 AUC | h24 bacc | h24 AUC | h48 bacc | h48 AUC |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| v1 / 2025 | 0.397295 | 0.608389 | 0.383024 | 0.583791 | 0.339534 | 0.534021 |
| audit_full / 2025 | 0.398615 | 0.611038 | 0.372716 | 0.591996 | 0.338316 | 0.552292 |
| audit_compact / 2025 | **0.401170** | **0.611966** | 0.369349 | 0.589047 | **0.344331** | 0.540842 |
| v1 / 2026 | **0.485388** | 0.649214 | 0.365542 | 0.603470 | **0.338060** | **0.606292** |
| audit_full / 2026 | 0.483429 | 0.648463 | 0.360219 | 0.613892 | 0.331717 | 0.574726 |
| audit_compact / 2026 | 0.480630 | **0.649270** | **0.426686** | **0.616559** | 0.329174 | 0.564596 |

## Decision

- `audit_compact` is the best candidate for h24 direction context. It raises 2026 h24 balanced accuracy from `0.365542` to `0.426686` and AUC from `0.603470` to `0.616559`.
- `audit_full` is not promoted. It improves some AUC values, but adds noise and weakens balanced accuracy.
- h12 remains close to baseline. `audit_compact` slightly improves h12 AUC in 2026, but lowers h12 balanced accuracy. Do not let h12 act as a standalone direction owner without downstream ablation.
- h48 should remain secondary/ranking context only. The v1 h48 AUC is still better on 2026.

## Artifacts

- `tmp/causal_regen_20260516/ai_patchmix_direction_core_audit_v2_20260530_full/`
- `tmp/causal_regen_20260516/ai_patchmix_direction_core_audit_compact_20260530_full/`
- comparison JSON: `tmp/causal_regen_20260516/ai_patchmix_direction_core_audit_compact_20260530_full/oos_compare_v1_full_compact.json`
