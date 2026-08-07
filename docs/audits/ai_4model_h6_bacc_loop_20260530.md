# AI 4-Model H6 BACC Loop - 2026-05-30

## Scope

User target: maximize AI direction accuracy, preferably `bacc >= 55%`, with h6 as the primary horizon because the current Regime3 stability/risk sidecar is h6.

Forbidden inputs remain:

- `teacher_*`
- `m7_*`
- `a5dir_*`
- `regime3_pred_*`
- Regime4 sidecars
- labels/targets/future/PnL-derived columns

Allowed regime context in this loop:

- split-local context: `cvp_regime`, `regime_trending`, `regime_persistence`
- Regime3 stability/risk h6 sidecar only: `regime3_stability_h6_score`, `regime3_transition_h6_risk_prob`, `regime3_transition_h6_risk_pred`, `regime3_churn_h6_risk_score`

## Artifacts

- PatchTSMixer h6 local-regime:
  - `tmp/causal_regen_20260516/ai_patchmix_h6_regime_local_20260530/`
- 4-family output head comparison:
  - `tmp/causal_regen_20260516/ai_4model_h6_bacc_max_20260530/summary.json`
- PatchTSMixer class-weight sweep:
  - `tmp/causal_regen_20260516/ai_patchmix_h6_classweight_sweep_20260530/summary.json`

## Results

2025 train -> 2026 OOS, h6 local triple-barrier labels.

| candidate | 2026 bacc | 2026 OVR AUC | note |
| --- | ---: | ---: | --- |
| PatchTSMixer h6 + local regime | **0.4865** | 0.6765 | best current bacc |
| TiDE family output head | 0.4703 | 0.6646 | useful risk context, not direction owner |
| old PatchTST family output head | 0.3301 | 0.5489 | poor h6 direct direction |
| DLinear family output head | 0.3367 | 0.5480 | poor h6 direct direction |
| all 4 AI output families | 0.4786 | 0.6319 | did not beat PatchTSMixer h6 |
| all 4 + compact context | 0.4776 | 0.6641 | did not beat PatchTSMixer h6 |

PatchTSMixer class-weight sweep:

| class weight power | 2026 bacc | 2026 OVR AUC |
| ---: | ---: | ---: |
| 0.00 | 0.4599 | 0.6751 |
| 0.25 | 0.4720 | 0.6761 |
| 0.50 | **0.4865** | 0.6765 |
| 0.75 | 0.4858 | **0.6776** |
| 1.00 | 0.4823 | 0.6682 |

## Decision

- Current best h6 bacc remains PatchTSMixer h6 + local regime with class-weight power `0.5`.
- Adding existing PatchTST/TiDE/DLinear output features does not improve h6 bacc.
- TiDE is still useful as risk/exit context, but not as the h6 direction owner.
- The bacc target `55%` was not reached in this loop.

## Next Loop

- Try label-boundary ablations for h6 rather than only larger model capacity.
- Test Chronos or TimesNet only if the runtime is feasible; do not block the h6 loop on slow TimesNet training.
- Keep `regime3_pred_*` removed unless the user explicitly reopens that contract.

## Follow-up Loop: Label Boundary + Chronos

Artifacts:

- Label sweep:
  - `scripts/sweep_ai_patchmix_h6_label_params_20260530.py`
  - `tmp/causal_regen_20260516/ai_patchmix_h6_label_sweep_20260530/summary.json`
- Chronos zero-shot h6:
  - `scripts/test_ai_chronos_h6_direction_20260530.py`
  - `tmp/causal_regen_20260516/ai_chronos_h6_direction_20260530/summary.json`
- Four-source Chronos stack:
  - `tmp/causal_regen_20260516/ai_4source_h6_chronos_stack_20260530/summary.json`

2025 train -> 2026 OOS, h6 labels.

| candidate | 2026 bacc | 2026 OVR AUC | note |
| --- | ---: | ---: | --- |
| PatchTSMixer `mae_light` label | 0.4971 | 0.6837 | best PatchTSMixer-only label sweep |
| Chronos h6 + core/local regime | 0.5009 | 0.6832 | best strict-clean source combo |
| Chronos h6 + core/local regime + TiDE outputs | **0.5020** | **0.6841** | research-only, because old TiDE artifact has timestamp gaps/NaNs |
| Chronos h6 only | 0.3672 | 0.5818 | weak alone |
| Chronos + PatchTSMixer h6 + core | 0.4951 | 0.6810 | did not improve |

Important contract note:

- `tmp/causal_regen_20260516/ai_feature_combo_grid/trade_candidates_2025_patchtst__tide__dlinear.csv` has 37 missing 2025 timestamps versus the year split and NaNs in old AI columns.
- Therefore old PatchTST/TiDE/DLinear outputs are research-only in this comparison. They are not active/live promotable until regenerated under the current fail-fast timestamp contract.

Interpretation:

- h6 signal improved from `0.4865` to `0.5009` with strict-clean Chronos+core.
- The 55% bacc target is still not reached.
- Feature importance for the best research stack is dominated by `atr14_pct`, then `realized_vol_24`, `cvp_volume_imbalance`, `ai_vol_regime_pct`, and `ret_6`; this suggests h6 labels are still heavily volatility-regime shaped, not clean directional prediction.

## Standalone Model Follow-up

User correction: do not rely on AI output ensembling; improve standalone model performance.

Standalone definition used here:

- no other AI model output columns;
- no `teacher_*`, `m7_*`, `a5dir_*`, `regime3_pred_*`, Regime4, future/target/PnL-derived columns;
- allowed inputs are the model's own representation plus direct current/core/local regime context.

Artifacts:

- Chronos/core standalone sweep:
  - `tmp/causal_regen_20260516/ai_single_model_h6_chronos_core_sweep_20260530/summary.json`
- Chronos/core standalone seed check:
  - `tmp/causal_regen_20260516/ai_single_model_h6_chronos_core_seedcheck_20260530/summary.json`

Best standalone single-seed candidate:

| model | label | 2026 bacc | 2026 OVR AUC | note |
| --- | --- | ---: | ---: | --- |
| Chronos h6 + core/local regime | `active_dense` | **0.5146** | 0.6590 | best single seed, lower ranking AUC |
| Chronos h6 + core/local regime | `mae_light` | 0.5034 | 0.6839 | better ranking AUC, weaker bacc |

Seed check, 5 seeds:

| model | mean bacc | std bacc | max bacc | mean AUC |
| --- | ---: | ---: | ---: | ---: |
| `active_dense_d5_l220_cw075` | **0.5114** | 0.0012 | 0.5132 | 0.6651 |
| `active_dense_d4_l224_cw075` | 0.5113 | 0.0013 | 0.5133 | 0.6655 |
| `mae_light_d4_l210_cw05` | 0.5013 | 0.0013 | 0.5034 | **0.6834** |

Decision:

- Current best standalone h6 bacc is Chronos h6 + core/local regime with `active_dense` labels and class-weight power `0.75`.
- It is stable across seeds around `0.511`.
- It is not a high-quality probability ranker; use it as a class/bias surface, not as a calibrated confidence source.
- `bacc >= 55%` remains unreached by standalone h6 models.

## Role-Specific TSFM Evaluation

User correction: do not force every AI family to be a standalone h6 direction classifier. Evaluate each family by its intended role.

Artifact:

- Runner: `scripts/run_ai_role_specific_experiments_20260530.py`
- Output: `tmp/causal_regen_20260516/ai_role_specific_eval_20260530/summary.json`
- 2025 features: `tmp/causal_regen_20260516/ai_role_specific_eval_20260530/tsfm_role_features_2025_exact.csv`
- 2026 features: `tmp/causal_regen_20260516/ai_role_specific_eval_20260530/tsfm_role_features_2026_exact.csv`
- Contract: exact timestamp regeneration, no cross-model ensembling for role metrics.
- All four local TSFM forecasters loaded: PatchTST, TiDE, DLinear, TimesNet.
- Warmup-only non-finite `tide_vol_zscore` values were set to `0.0` and recorded in the manifests. Timestamp gaps remain fail-fast.

Role metrics, 2026 OOS:

| family / role | metric | result | interpretation |
| --- | --- | ---: | --- |
| PatchTST direction | h6 bacc | 0.3452 | not usable as standalone h6 direction owner |
| PatchTST direction | h12 bacc | 0.3475 | not usable as standalone h12 direction owner |
| PatchTST direction | h6 up/down AUC | 0.4896 / 0.5027 | no useful directional ranking |
| Chronos distribution | h6 q50-sign bacc | 0.3426 | q50 sign is not a direction owner |
| Chronos distribution | median return corr | 0.0100 | weak central-path signal |
| Chronos distribution | large-move AUC | 0.5511 | modest large-move context only |
| TiDE risk | h6 top30 adverse AUC raw | 0.7354 | strong adverse-risk / exit / sizing context |
| TiDE risk | h12 top30 adverse AUC raw | 0.7227 | strong adverse-risk / exit / sizing context |
| TiDE risk | h6 adverse corr raw | 0.3640 | useful risk magnitude signal |
| DLinear trend/flow | h24 trend AUC flow | 0.4938 | weak trend classifier |
| DLinear trend/flow | h24 return corr flow | 0.0469 | small low-frequency drift context |
| TimesNet cycle | entry quality AUC anchor revert | 0.5193 | weak but possible session/cycle context |
| TimesNet cycle | cycle delta return corr | -0.0237 | weak direct edge |

Decision:

- Do not promote PatchTST/PatchTSMixer or Chronos q50 sign as hard standalone h6/h12 direction owners from this role test.
- Promote TiDE-derived outputs as risk-layer candidates only: adverse-risk veto, exit pressure, notional/leverage downsize, TP/SL widening/narrowing.
- Keep DLinear as optional low-frequency drift/flow context only after downstream ablation.
- Keep TimesNet as optional session/cycle context only after downstream ablation.
- The best standalone h6 class-bias result remains Chronos/core `active_dense` from the seed check, not the raw q50 sign role test.

## Reworked Input Retrain

User request: TiDE was the only clearly useful role model, so re-analyze the inputs and retrain/test the other AI routes.

Artifacts:

- Reworked NF runner: `scripts/retrain_ai_role_models_reworked_inputs_20260530.py`
- Reworked NF output: `tmp/causal_regen_20260516/ai_role_models_reworked_inputs_20260530/summary.json`
- Reworked PatchTSMixer output: `tmp/causal_regen_20260516/ai_patchmix_h6_reworked_inputs_20260530/summary.json`

Training contract:

- NF TiDE/DLinear reworked candidates are trained on `data/splits/year_oos/training_features_2024.csv` only.
- They are exact-timestamp scored on 2025 and 2026.
- PatchTSMixer route uses `fit2024 -> score2025` and `fit2025 -> score2026`.
- A strict `fit2024 -> score2026` PatchTSMixer check was also run after the user asked whether the AI was trained on 2024.
- Existing `data/nf_*` live packs were not overwritten.

Input changes:

- PatchTSMixer direction route:
  - compact audited directional features plus local regime context.
  - h6/h12 heads added.
- TiDE risk route:
  - volatility/compression, clean funding/OI, crowding/squeeze, CVP, and local regime context.
- DLinear flow route:
  - flow/CVD/taker/whale/BTC-lead pressure context.
- TimesNet:
  - full CPU retrain was started but was too slow for this loop. It remains separated until a cheaper TimesNet-specific setup is used.

2026 OOS comparison:

| route | old role result | reworked result | decision |
| --- | ---: | ---: | --- |
| PatchTSMixer/PatchTST h6 direction bacc | 0.3452 raw PatchTST role output | **0.5016** PatchTSMixer+CatBoost h6 | improved, but confidence AUC is still weak |
| PatchTSMixer/PatchTST h12 direction bacc | 0.3475 raw PatchTST role output | **0.4983** PatchTSMixer+CatBoost h12 | improved, not hard owner |
| PatchTSMixer strict `2024->2026` h6 bacc | n/a | **0.5079** | better strict check, still below Chronos/core seed mean |
| PatchTSMixer strict `2024->2026` h12 bacc | n/a | **0.4821** | weaker than h6; secondary context only |
| TiDE h6 adverse AUC raw | 0.7354 | **0.7484** | stronger risk-layer candidate |
| TiDE h12 adverse AUC raw | 0.7227 | **0.7336** | stronger risk-layer candidate |
| DLinear h24 trend AUC flow | 0.4938 | 0.4929 | no meaningful improvement |
| DLinear h24 return corr flow | 0.0469 | 0.0472 | tiny drift context only |

Decision:

- TiDE remains the only clearly strong AI route, and reworked inputs improved it.
- PatchTSMixer is no longer useless after input rework, but it is still below the best standalone Chronos/core h6 class-bias model (`0.5114` mean bacc).
- PatchTSMixer h12 values above are evaluated with the actual `ai_patch_h12_*` head. Earlier scratch output that reused h6 predictions for h12 is superseded.
- DLinear should not be promoted as an entry/trend owner.
- TimesNet needs a separate lightweight cycle/session experiment; do not block the main AI feature loop on full CPU TimesNet training.

## Chronos Standalone Multi-Series / Derived-Series Test

User constraint: do not use option 3 / downstream CatBoost or meta heads for Chronos. Improve the AI model standalone through:

1. multi-series Chronos inputs
2. derived time-series Chronos inputs

Artifact:

- Runner: `scripts/test_chronos_multiseries_standalone_20260530.py`
- Summary: `tmp/causal_regen_20260516/chronos_multiseries_standalone_20260530/summary.json`

Contract:

- Chronos zero-shot / standalone only.
- No CatBoost, no downstream meta head, no cross-model ensemble.
- Threshold and inversion are selected on 2025 and then fixed for 2026.
- Tested series: `log_close`, `ret6_z`, `flow_pressure`, `funding_pressure`, `cvd_288`, `price_cvd_divergence`, `vwap_dist_96`, `range_breakout`.

Best 2026 OOS results after 2025 threshold/inversion selection:

| series | accuracy | bacc | up AUC | down AUC | large-move AUC | decision |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `price_cvd_divergence` | 0.3108 | **0.3853** | 0.4960 | 0.5027 | **0.6539** | not a direction owner; useful large-move context candidate |
| `vwap_dist_96` | 0.2990 | 0.3770 | 0.4905 | 0.4938 | 0.6402 | not a direction owner; useful large-move context candidate |
| `flow_pressure` | 0.3110 | 0.3718 | 0.4853 | 0.5191 | 0.4365 | reject as standalone Chronos direction route |
| `ret6_z` | 0.3043 | 0.3658 | 0.4861 | 0.5037 | 0.5960 | reject as standalone Chronos direction route |
| `log_close` | 0.4170 | 0.3588 | 0.4797 | 0.5393 | 0.5494 | reject as standalone Chronos direction route |
| `cvd_288` | 0.3304 | 0.3526 | 0.4869 | 0.4895 | 0.5502 | reject as standalone Chronos direction route |
| `range_breakout` | 0.2826 | 0.3513 | 0.5175 | 0.4592 | 0.5277 | reject as standalone Chronos direction route |
| `funding_pressure` | 0.4076 | 0.3497 | 0.4735 | 0.5146 | 0.5035 | reject as standalone Chronos direction route |

Decision:

- Chronos standalone multi-series / derived-series tests failed as h6 direction owners. The best bacc is only `0.3853`.
- Do not promote Chronos standalone output into active direction logic.
- `price_cvd_divergence` and `vwap_dist_96` Chronos forecast width / movement magnitude may still be tested as uncertainty or large-move context, but only as a downstream ablation and not as a hard entry owner.
- This result supersedes any idea that Chronos can be fixed for direction by series selection alone.

TimesNet status:

- Background TimesNet reworked-input run directory: `tmp/causal_regen_20260516/ai_timesnet_reworked_inputs_bg_20260530/`
- The process is no longer running.
- No summary artifact was produced; only `run.log` exists and stops around epoch 48.
- Treat this as incomplete/no-result, not as a failed metric result.

## PatchTSMixer Binary Tradeable Long/Short Target


Artifacts:

- Runner: `scripts/train_ai_patchmix_binary_tradeable_20260530.py`
- Summary: `tmp/causal_regen_20260516/ai_patchmix_binary_tradeable_20260530/summary.json`
- Log: `tmp/causal_regen_20260516/ai_patchmix_binary_tradeable_20260530_run.log`

Feature contract:

- Reused the already expanded `audit_compact_local_regime` PatchTSMixer input.
- Inputs include the compact audited directional/context set plus local non-pred regime context.
- No `teacher_*`, `m7_*`, `a5dir_*`, existing `ai_*`, labels, targets, future path, or PnL-derived columns are included.

Target contract:

- Binary target only: `0=short`, `1=long`.
- Neutral/flat bars are excluded from the binary training and evaluation target.
- A bar is tradeable only if one side clears the dynamic edge floor and beats the opposite side by the configured margin.
- Configs tested: `tradeable_dense`, `tradeable_base`, `tradeable_fee2`, `tradeable_high_quality`.

Best results:

| split | horizon | best config | bacc | AUC | accuracy | tradeable coverage |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| `2024->2025` | h6 | `tradeable_high_quality` | **0.5198** | 0.5282 | 0.5200 | 0.5605 |
| `2024->2025` | h12 | `tradeable_high_quality` | 0.5123 | 0.5165 | 0.5125 | 0.7073 |
| `2025->2026` | h6 | `tradeable_dense` | **0.5133** | 0.5097 | 0.5125 | 0.8412 |
| `2025->2026` | h12 | `tradeable_base` | 0.4911 | 0.4835 | 0.4907 | 0.8619 |
| strict `2024->2026` | h6 | `tradeable_fee2` | **0.5249** | **0.5368** | 0.5248 | 0.6166 |
| strict `2024->2026` | h12 | `tradeable_fee2` | 0.5192 | 0.5293 | 0.5166 | 0.7568 |


- New strict `2024->2026` h6 binary tradeable bacc: `0.5249`
- New strict `2024->2026` h12 binary tradeable bacc: `0.5192`

Decision:

- The strongest candidate is h6 `tradeable_fee2` from strict `2024->2026`; it improves both bacc and AUC and keeps a usable tradeable coverage of `0.6166`.
- Do not promote it as a hard standalone entry owner yet. It should be tested next as an Alpha6/Alpha7 entry-context feature or direction-bias feature in PnL/MDD/trade-count backtests.
- h12 improved in strict mode but failed on `2025->2026`; use h12 as secondary context only.

## TimesNet Role Lock: Session / Anchor Reversion Context

User decision: fix TimesNet as a session / anchor-reversion auxiliary feature family, not a direction owner.

Latest completed artifact:

- Output directory: `tmp/causal_regen_20260516/ai_timesnet_direction_inputs_bg_20260530/`
- Summary: `tmp/causal_regen_20260516/ai_timesnet_direction_inputs_bg_20260530/summary.json`
- Feature CSVs:
  - `tmp/causal_regen_20260516/ai_timesnet_direction_inputs_bg_20260530/role_features_2025_reworked.csv`
  - `tmp/causal_regen_20260516/ai_timesnet_direction_inputs_bg_20260530/role_features_2026_reworked.csv`
- Contract: `tmp/causal_regen_20260516/ai_timesnet_direction_inputs_bg_20260530/nf_timesnet/reworked_input_contract.json`

Inputs:

- The reworked TimesNet run uses 31 inputs: time/session, VWAP anchor distance, compression/ATR/range/wick, CVP/local regime, and added direction/flow/funding/BTC-lead context.
- Target remains `y_vwap_dev`.
- Outputs remain:
  - `ai_anchor_revert_prob`
  - `ai_anchor_overheat`
  - `ai_anchor_trend_escape_prob`
  - `timesnet_cycle_sin`
  - `timesnet_cycle_cos`
  - `timesnet_cycle_delta`

2026 OOS role metrics:

| metric | value |
| --- | ---: |
| `entry_quality_auc_anchor_revert` | `0.51996` |
| `entry_quality_auc_trend_escape` | `0.48004` |
| `cycle_delta_ret_corr` | `-0.02176` |

Decision:

- Do not train or evaluate TimesNet as a hard long/short direction owner in this lineage.
- Do not use TimesNet output as an entry direction signal.
- Use TimesNet only as auxiliary context for session/anchor-reversion behavior:
  - raise/lower entry threshold around anchor-reversion regimes;
  - reduce notional/leverage when `ai_anchor_overheat` is high;
  - prefer shorter TP / faster exit when reversion probability is high;
  - veto mean-reversion entries when `ai_anchor_trend_escape_prob` is high.
- Downstream promotion still requires PnL/MDD/trade-count ablation.

## Role-Based Pass Reassessment

User correction: AI features do not all need to be directional. Re-evaluate by each model family's intended trading role.

Artifact:

- `tmp/causal_regen_20260516/ai_role_pass_reassessment_20260530.json`

Pass criteria used here:

- Direction-bias context: bacc/AUC must improve over the previous same-family direction baseline and keep usable coverage.
- Risk/exit/sizing context: risk/adverse AUC or correlation must be clearly useful even if direction bacc is weak.
- Uncertainty/large-move context: large-move AUC can pass even when long/short direction fails.
- Session/anchor context: weak positive signal can pass only as a small modifier, not as a gate.

| family | intended role | status | evidence | allowed use |
| --- | --- | --- | --- | --- |
| TiDE | adverse-risk / exit / sizing | **PASS** | h6 adverse AUC raw `0.7484`, h12 `0.7336` | risk veto, notional/leverage resize, exit pressure |
| PatchTSMixer binary | tradeable h6 long/short direction bias | **HOLD_FAIL** | strict `2024->2026` h6 bacc `0.5249`, AUC `0.5368`, but `2025->2026` h6 bacc/AUC degraded to about `0.5133`/`0.5097` | do not promote unless a later input/label redesign fixes OOS instability |
| Chronos | large-move / uncertainty context | **PASS** | expanded uncertainty retest with live-safe EWM smoothing: `atr14_pct` `upside_band_ewm3` gives 2025 large/downside AUC `0.6050`/`0.6018`, 2026 `0.6228`/`0.6307`; direction bacc fails | uncertainty / large-move / downside-risk modifier only |
| TimesNet | anchor reversion / overheat / session context | **WEAK_PASS_CANDIDATE** | anchor-revert AUC `0.5200`; trend-escape and cycle edge weak | small threshold/size/exit modifier only |
| DLinear | low-frequency flow/trend drift | **HOLD_FAIL** | h24 trend AUC `0.4929`; ret corr `0.0472` | no promotion unless downstream ablation proves value |

Updated interpretation:

- TiDE is not the only usable AI family if role-specific use is allowed.
- TiDE remains the only strong pass.
- PatchTSMixer binary, Chronos, and TimesNet are context candidates with strict usage limits.
- DLinear remains held out.
- None of the context candidates should create new entries by themselves. The next valid test is to inject them into Alpha6/Alpha7 as modifiers and evaluate PnL/MDD/trades.

## Chronos Expanded Uncertainty / Large-Move Retest

User request: change Chronos role and expand input series. Chronos is not locally fine-tuned in this repo; this retest regenerates zero-shot quantile forecasts with the new role contract.

Artifacts:

- Runner: `scripts/test_chronos_uncertainty_large_move_20260530.py`
- Summary: `tmp/causal_regen_20260516/chronos_uncertainty_large_move_20260530/summary.json`
- Log: `tmp/causal_regen_20260516/chronos_uncertainty_large_move_20260530_run.log`

Contract:

- Chronos output is not a long/short direction owner.
- Inputs are expanded live-safe current/past series: price/return, flow, funding/OI, squeeze/crowding, CVD divergence, VWAP distance, compression/breakout, ATR and realized volatility.
- Evaluation target is h6 large move / downside risk, not h6 direction bacc.
- No downstream CatBoost/meta head and no validation-distribution thresholding is used.

2026 OOS top scores:

| input series | Chronos score | large-move AUC | downside AUC | top10 large-move lift | top10 downside lift |
| --- | --- | ---: | ---: | ---: | ---: |
| `atr14_pct` | `upside_band_ewm3` | **0.6228** | **0.6307** | 1.572 | 1.497 |
| `atr14_pct` | `width` | 0.6172 | 0.6188 | 1.426 | 1.509 |
| `realized_vol_24` | `width` | 0.6152 | 0.6039 | 1.438 | 1.397 |
| `atr14_pct` | `large_move_score` | 0.6124 | 0.6077 | 1.590 | 1.578 |
| `realized_vol_24` | `large_move_score` | 0.6106 | 0.6003 | 1.497 | 1.460 |
| `atr14_pct` | `upside_band` | 0.6094 | 0.6156 | 1.596 | 1.523 |

Decision:

- Chronos is upgraded to **PASS** as an uncertainty / large-move / downside-risk context feature family.
- The best practical feature candidate is `chronos_atr14_upside_band_ewm3`.
- Backup candidates are `chronos_atr14_width_ewm6`, `chronos_atr14_width`, `chronos_atr14_large_move_score`, `chronos_realized_vol24_width`, and `chronos_realized_vol24_large_move_score`.
- Do not use Chronos as an entry direction owner. Downstream use should be risk resize, entry threshold tightening, TP/SL widening, or exit pressure when projected uncertainty is high.

## PatchTST Tradeable Representation Test

User request: test PatchTST as an alternative to PatchTSMixer.

Artifact:

- Runner: `scripts/train_ai_patchtst_tradeable_20260530.py`
- Summary: `tmp/causal_regen_20260516/ai_patchtst_tradeable_20260530/summary.json`
- Log: `tmp/causal_regen_20260516/ai_patchtst_tradeable_20260530_run.log`

Contract:

- PatchTST was trained from scratch because no local PatchTST HF pretrained checkpoint is available.
- Input contract matches the PatchTSMixer binary experiment: `audit_compact_local_regime` patch channels.
- Label is h6 `tradeable_fee2` binary short/long.
- Strict comparison is `2024 -> 2026`.
- Compared three heads:
  - PatchTST end-to-end classifier
  - trained PatchTST encoder embedding + MLP
  - trained PatchTST encoder embedding + CatBoost

2026 OOS result:

| variant | bacc | AUC | accuracy | decision |
| --- | ---: | ---: | ---: | --- |
| PatchTST end-to-end | **0.5050** | 0.5054 | 0.5021 | reject |
| PatchTST embedding + MLP | 0.5009 | 0.5002 | 0.5029 | reject |
| PatchTST embedding + CatBoost | 0.5046 | **0.5080** | 0.5041 | reject |

Comparison:

- PatchTSMixer binary strict h6 `tradeable_fee2`: bacc `0.5249`, AUC `0.5368`.
- PatchTST did not come close to PatchTSMixer under the same label/input family.

Decision:

- Do not replace PatchTSMixer binary with PatchTST.
- PatchTST is not promoted as a PASS/PASS_CANDIDATE in the current AI map.
- A future PatchTST test would need either a real pretrained local checkpoint or a different pretraining objective; training from scratch is not sufficient here.
