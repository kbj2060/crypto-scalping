# Regime3 Current HMM Wide24 Experiment - 2026-05-29

## Status

Confirmed next Regime3 CURRENT research candidate. Not wired into live trading.

## Purpose

Compare the original compact `state12` Regime3 current HMM with a wider 24-feature current HMM.

## Feature Sets

`state12` uses:

- `state7_trend_score`
- `state7_trend_efficiency_48`
- `state7_directional_return_48`
- `state7_volatility_state`
- `state7_sign_flip_rate_24`
- `state7_range_compression`
- `state7_flow_alignment`
- `state12_log_return`
- `state12_garman_klass_vol`
- `state12_net_taker_ratio`
- `state12_oi_change_rate`
- `state12_chop_index`

`wide24` adds:

- `volatility_z`
- `rsi`
- `macd_hist`
- `bb_width_z`
- `hma_slope`
- `wick_ratio`
- `mtf_trend_1h`
- `mtf_trend_4h`
- `breakout_strength`
- `mean_reversion_z`
- `ofi_acceleration`
- `taker_acceleration`

## Results

2026 forward test:

| feature set | accuracy | balanced accuracy | bull recall | bear recall | chop recall | flip rate | mean duration bars |
|---|---:|---:|---:|---:|---:|---:|---:|
| `state12` | 0.7305 | 0.7655 | 0.7951 | 0.7819 | 0.7195 | 0.0417 | 23.93 |
| `wide24` | 0.7698 | 0.8143 | 0.8943 | 0.7919 | 0.7567 | 0.0520 | 19.22 |
| `docs42` | 0.7263 | 0.8153 | 0.9415 | 0.8052 | 0.6992 | 0.0944 | 10.59 |
| `docs51all` | 0.7343 | 0.8307 | 0.9138 | 0.8743 | 0.7038 | 0.1077 | 9.28 |
| `docs51all sticky97` | 0.7337 | 0.8302 | 0.9138 | 0.8737 | 0.7032 | 0.1077 | 9.28 |

## Docs Feature Audit Follow-Up

Tested Docs Manager feature-audit guided CURRENT feature packs:

- `docs42`: `wide24` plus audit-approved directional/context features such as `compression_score`, `atr_pct_rank_288`, `bb_width_pct_rank_288`, `btc_volume_impulse_z`, `vwap_dist_*`, `cvd_*`, `eth_btc_ret_spread_*`, `price_cvd_divergence`, `crowding_pressure`, and `long_squeeze_risk`.
- `docs51all`: `docs42` plus clean-funding and raw volume risk-context columns.

Result:

- `docs51all` has the highest 2026 balanced accuracy.
- Both docs packs materially increase flip rate and shorten mean state duration.
- Raising sticky to `0.97` did not reduce `docs51all` churn.

Decision:

- Keep `wide24` as the main Regime3 CURRENT research surface.
- Keep `docs51all` as a high-sensitivity research variant only.
- Do not feed `docs51all` into PRED or downstream action models without a churn-tolerance/backtest check.

## Interpretation

`wide24` improves balanced accuracy and all class recalls on 2026, especially bull recall.

The cost is a higher flip rate and shorter average regime duration. This is probably acceptable for feature context, but should not be promoted to live action ownership without checking downstream trading sensitivity to extra regime churn.

Decision: use `wide24` as the next Regime3 CURRENT surface for follow-on Regime3 PRED/risk experiments.

## Artifacts

- Script: `scripts/experiment_regime3_current_hmm_wide24_20260529.py`
- Report: `data/ensemble/reports/regime3_current_hmm_wide24_experiment_20260529_report.json`
- Artifact directory: `data/ensemble/supervised/regime3_current_hmm_wide24_experiment_20260529/`

## Sensitive Wide24 Promotion - 2026-05-30

User-selected next Regime3 CURRENT surface:

- Model ID: `regime3_current_hmm_sensitive_balancedish_20260530`
- Label mode: `balancedish_adx16_slope15_bb012`
- CURRENT prefix: `regime3_current_sensitive_wide24_`
- Sidecar stem: `regime3_current_sensitive_hmm_wide24`
- Artifact directory: `data/ensemble/supervised/regime3_current_hmm_sensitive_balancedish_20260530/`
- Selection manifest: `data/ensemble/supervised/regime3_current_hmm_sensitive_balancedish_20260530/SELECTED_MAIN_REGIME3_CURRENT_20260530.json`

Label thresholds:

- `trend_adx_min = 16.0`
- `weak_adx_max = 12.0`
- `slope_min = 0.00015`
- `tight_bb_max = 0.012`

2026 OOS metrics for `wide24`:

| accuracy | balanced accuracy | bull recall | bear recall | chop recall | flip rate | mean duration bars |
|---:|---:|---:|---:|---:|---:|---:|
| 0.7173 | 0.7740 | 0.8042 | 0.8479 | 0.6701 | 0.1275 | 7.84 |

This variant is intentionally more sensitive than the original `wide24` surface. It reduces target-label chop dominance, but materially increases regime churn. Downstream PRED/action models must use the explicit `regime3_current_sensitive_wide24_*` contract; do not alias it to `regime3_current_wide24_*`.

Follow-on PRED retrain:

- Model ID: `regime3_pred_tft_vsn_sensitive_wide24_current_docs48all_20260530`
- PRED prefix: `regime3_pred_sensitive_tft_h12_`
- Feature pack: `docs_regime_pred_all`
- Selected feature count: `48`
- Current feature count: `9`
- Artifact directory: `data/ensemble/supervised/regime3_pred_tft_vsn_sensitive_wide24_current_docs48all_20260530/`
- Selection manifest: `data/ensemble/supervised/regime3_pred_tft_vsn_sensitive_wide24_current_docs48all_20260530/SELECTED_MAIN_REGIME3_PRED_20260530.json`

2026 PRED OOS metrics:

| accuracy | balanced accuracy | bull recall | bear recall | chop recall | log loss |
|---:|---:|---:|---:|---:|---:|
| 0.5906 | 0.5760 | 0.4975 | 0.6018 | 0.6285 | 0.8999 |

The PRED input feature family is the previous Docs48 all-raw design with only the CURRENT contract changed from `regime3_current_wide24_*` to `regime3_current_sensitive_wide24_*`. The target also changes accordingly because PRED labels are `argmax(current at t+horizon)`.

## Sensitive PRED Improvement Sweep - 2026-05-30

After the sensitive CURRENT promotion, PRED was retested with larger TFT hidden size and expanded input packs while keeping the new CURRENT contract fixed.

Horizon `h=12` remains difficult because the sensitive CURRENT surface has higher churn. A one-hour-ahead persistence baseline already sits near the learned model, so feature/parameter scaling only produced marginal improvement.

| candidate | horizon bars | feature pack | seq len | features | 2026 acc | 2026 bacc | 2026 log loss |
|---|---:|---|---:|---:|---:|---:|---:|
| `regime3_pred_sensitive_rolled_top72_d96_e10_20260530` | 12 | `docs_regime_pred_rolled` | 72 | 66 | 0.6281 | 0.5809 | 0.8571 |
| `regime3_pred_tft_vsn_sensitive_wide24_current_docs48all_20260530` | 12 | `docs_regime_pred_all` | 72 | 48 | 0.5906 | 0.5760 | 0.8999 |
| `regime3_pred_sensitive_docsall_top48_d96_e10_20260530` | 12 | `docs_regime_pred_all` | 72 | 48 | 0.5747 | 0.5745 | 0.9351 |
| `regime3_pred_sensitive_default_top96_d96_e10_20260530` | 12 | `default` | 72 | 96 | 0.5515 | 0.5379 | 0.9440 |
| `regime3_pred_sensitive_h6_docsall_top48_d96_e10_20260530` | 6 | `docs_regime_pred_all` | 72 | 48 | 0.6857 | 0.6774 | 0.7368 |
| `regime3_pred_sensitive_h6_rolled_top72_d96_e10_20260530` | 6 | `docs_regime_pred_rolled` | 72 | 66 | 0.6795 | 0.6713 | 0.7370 |

Best `h=12` candidate:

- `data/ensemble/supervised/regime3_pred_sensitive_rolled_top72_d96_e10_20260530/CANDIDATE_BEST_H12_RESEARCH_CANDIDATE_20260530.json`

Best `h=6` candidate:

- `data/ensemble/supervised/regime3_pred_sensitive_h6_docsall_top48_d96_e10_20260530/CANDIDATE_BEST_H6_RESEARCH_CANDIDATE_20260530.json`

Decision note:

- Keep `h=12` only if one-hour-ahead medium-horizon context is mandatory.
- Prefer testing `h=6` for downstream routing/risk context because the sensitive CURRENT regime surface is too churn-heavy for stable one-hour prediction.

## No-Current PRED Shortcut Test - 2026-05-30

The first sensitive PRED models used `regime3_current_sensitive_wide24_*` as input features and were found to strongly match same-bar CURRENT (`shift +0`) rather than only the intended future target. To test shortcut leakage through CURRENT inputs, PRED was retrained with CURRENT columns removed from the feature set while still using CURRENT only as the target source.

Best no-current candidate:

- Model ID: `regime3_pred_sensitive_h6_nocurrent_rolled_top72_d96_e10_20260530`
- PRED prefix: `regime3_pred_sensitive_nocur_tft_h6_`
- Feature pack: `docs_regime_pred_rolled`
- Selected feature count: `57`
- Current feature count: `0`
- Manifest: `data/ensemble/supervised/regime3_pred_sensitive_h6_nocurrent_rolled_top72_d96_e10_20260530/CANDIDATE_BEST_NO_CURRENT_H6_PRED_20260530.json`

2026 OOS comparison:

| candidate | current features | 2026 acc | 2026 bacc | transition bacc | persistence bacc |
|---|---:|---:|---:|---:|---:|
| `h6_docsall_top48` | 9 | 0.6857 | 0.6774 | 0.2023 | 0.9233 |
| `h6_nocurrent_rolled_top72` | 0 | 0.6846 | 0.6695 | 0.2534 | 0.8882 |

Interpretation:

- Removing CURRENT features slightly lowers total OOS balanced accuracy.
- It improves transition-row balanced accuracy from `0.2023` to `0.2534`, confirming that the original model had a CURRENT-copy shortcut.
- For actual transition-risk use, prefer the no-current candidate or train a dedicated transition/hazard head instead of using same-bar CURRENT-conditioned PRED directly.

## Dedicated Transition Hazard Head - 2026-05-30

Trained a dedicated h6 transition system instead of forcing the future-regime PRED head to solve both persistence and transition:

- Hazard head: binary `current[t] != current[t+6]`
- Destination head: future regime class, trained on transition rows
- Combination rule: if hazard probability exceeds threshold, use destination head with the current class masked out; otherwise keep current regime

Best operational research candidate:

- Model ID: `regime3_transition_hazard_sensitive_h6_20260530`
- Artifact directory: `data/ensemble/supervised/regime3_transition_hazard_sensitive_h6_withcurrent_20260530/`
- Manifest: `data/ensemble/supervised/regime3_transition_hazard_sensitive_h6_withcurrent_20260530/CANDIDATE_TRANSITION_HAZARD_H6_WITHCURRENT_THR046_20260530.json`
- Selected threshold: `0.46`
- Feature count: `66`
- CURRENT input features: enabled as from-state context, not as a future-class shortcut

2026 OOS:

| model | overall bacc | transition bacc | persistence bacc | hazard precision | hazard recall | fire rate |
|---|---:|---:|---:|---:|---:|---:|
| h6 future PRED with CURRENT input | 0.6774 | 0.2023 | 0.9233 | n/a | n/a | n/a |
| h6 no-current future PRED | 0.6695 | 0.2534 | 0.8882 | n/a | n/a | n/a |
| transition hazard/destination `thr=0.46` | 0.5881 | 0.5163 | 0.6386 | 0.4939 | 0.6969 | 0.4512 |

Interpretation:

- The transition head more than doubles transition-row balanced accuracy versus the original future-regime PRED.
- It is not a replacement for the full PRED class head because overall/persistence accuracy drops.
- Use it as a transition-risk/hazard context or veto/sizing input, not as the sole regime classifier.

## Stable H6 Decoder Without CURRENT Probability Inputs - 2026-05-30

User constraint: do not use `CURRENT stable probs` or `CURRENT sensitive probs` as model input features.

Implemented a no-current stable decoder:

- Script: `scripts/train_regime3_stable_decoder_20260530.py`
- Artifact directory: `data/ensemble/supervised/regime3_stable_h6_decoder_nocurrent_transitionaware_20260530/`
- Manifest: `data/ensemble/supervised/regime3_stable_h6_decoder_nocurrent_transitionaware_20260530/CANDIDATE_STABLE_H6_DECODER_NO_CURRENT_TRANSITIONAWARE_20260530.json`
- Model inputs: raw/docs rolled features only; `current_feature_count = 0`
- CURRENT sidecar use: label generation and evaluation only
- Stable target: smoothed sensitive CURRENT argmax with selected `min_duration = 6`
- Selected mode: direct multiclass decoder
- Transition weighting: `2.0`

2026 OOS selected result:

| overall acc | overall bacc | transition bacc | persistence bacc | transition rows | persistence rows |
|---:|---:|---:|---:|---:|---:|
| 0.7583 | 0.7410 | 0.2773 | 0.8403 | 2640 | 14251 |

The no-current decoder is cleaner than current-conditioned PRED, but transition bacc remains modest. It should be considered a cleaner full-regime classifier candidate, not a solved transition predictor.

## Stability/Risk Feature Head - 2026-05-30

Reframed Regime3 PRED away from directional/future-class prediction and into risk/stability features.

User constraint remains active:

- No `CURRENT stable probs` as model inputs.
- No `CURRENT sensitive probs` as model inputs.
- Current sidecar is used only for label generation/evaluation.

Candidate:

- Script: `scripts/train_regime3_stability_risk_20260530.py`
- Model ID: `regime3_stability_risk_h6_20260530`
- Artifact directory: `data/ensemble/supervised/regime3_stability_risk_h6_20260530/`
- Manifest: `data/ensemble/supervised/regime3_stability_risk_h6_20260530/CANDIDATE_STABILITY_RISK_H6_NO_CURRENT_20260530.json`
- Feature count: `90`
- CURRENT feature count: `0`

Output columns:

- `regime3_stability_h6_score`
- `regime3_transition_h6_risk_prob`
- `regime3_transition_h6_risk_pred`
- `regime3_churn_h6_risk_score`

2026 OOS risk separation:

| metric | value |
|---|---:|
| transition AUC | 0.6762 |
| transition bacc at validation threshold | 0.5872 |
| top 20% risk transition rate | 0.2874 |
| low 20% risk transition rate | 0.0471 |
| risk score correlation with target | 0.2977 |

Interpretation:

- This head should not be used as a hard directional regime predictor.
- It is useful as a continuous risk-rank / size-throttle / veto context.
- Suggested 2026 top-risk cutoffs: `transition_prob >= 0.5935` or `churn_score >= 0.2400`.

## Active Policy Update - 2026-05-30

Directional `PRED regime` is removed from active action ownership.

Active downstream usage:

- Keep CURRENT regime as `regime3_current_sensitive_wide24_*`.
- Use `regime3_stability_h6_score`, `regime3_transition_h6_risk_prob`, `regime3_transition_h6_risk_pred`, and `regime3_churn_h6_risk_score` as stability/transition-risk features.
- Use those risk features for veto, size throttle, leverage reduction, TP/SL/hold tightening, and uncertainty context only.

Forbidden downstream usage:

- Do not use `regime3_pred_*` future class probabilities as long/short direction.
- Do not use `regime3_pred_*` as primary/fallback action labels.
- Do not use `regime3_pred_*` as hard future regime selectors in active candidate paths.
- Do not silently alias old PRED contracts into the current active contract.

See `docs/active_live/regime3_policy_20260530.md` for the concise active policy.
