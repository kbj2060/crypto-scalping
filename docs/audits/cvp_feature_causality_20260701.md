# CVP Feature Causality Audit - 2026-07-01

- Verdict: `CVP_FEATURE_CAUSALITY_PASS`
- Source check pass: `True`

## Prefix Stability

| CSV | Pass | Prefix Rows | Extended Rows | Max Diff |
| --- | --- | ---: | ---: | ---: |
| `/home/llewyn/crypto-scalping/tmp/causal_regen_20260516/alpha5_a5dir_2024_train_2025_score_20260529_cleanfunding_stable48/02_fixed_regime4_state24_sticky090_tp18_sl10_preprocess_2024_to_2025/trade_candidates_2025_regime4_state24_sticky090_tp18_sl10_fixed.csv` | `True` | `350` | `550` | `0.000e+00` |
| `/home/llewyn/crypto-scalping/tmp/causal_regen_20260516/alpha7_01965_cleanfunding_candidates_20260529/trade_candidates_2026_alpha6_current_tail111_exact.csv` | `True` | `350` | `550` | `0.000e+00` |

## Source Evidence

- add_cvp_features slices each row as start:i+1, so each output row uses only current and prior bars.

## Artifacts

- JSON: `/home/llewyn/crypto-scalping/tmp/causal_regen_20260516/cvp_feature_causality_20260701/cvp_feature_causality_20260701.json`
