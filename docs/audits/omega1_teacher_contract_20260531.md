# Omega1 Teacher Contract - 2026-05-31

## Scope

Omega1 is the current teacher-stack planning line. Its allowed second-stage
families are:

- Regime3 stability/risk h6 sidecar
- M7 scored outputs
- AI/TSFM scored outputs
- Omega1 teacher outputs derived from those families

Regime4 is not part of Omega1 teacher inputs.

## Hard Exclusions

The Omega1 teacher input contract must reject all Regime4 families:

- `clean_regime4_state24_sticky090_v2_*`
- `clean_regime4_2024_unsup_v1_*`
- `clean_regime_2024_unsup_v4_*`
- `regime4_pred_*`

Omega1 teacher builders must also reject:

- `teacher_*` feedback inputs
- `a5dir_*` router outputs
- labels, targets, realized PnL, future path statistics, and action-score
  columns

`teacher_*` columns are allowed only after the teacher layer has emitted them,
as inputs to downstream Omega1 parent/risk/final-policy models. They must never
be used as inputs to AI, M7, Regime3, router, or teacher-generation jobs unless
a new out-of-fold/no-leak stacking contract is created.

## Canonical Action Label Contract

Omega1 active teacher training must use the 3-class ZigZag action label
dataset as the canonical action label source.

Label semantics:

- `0`: CASH / no-trade state
- `1`: LONG ZigZag segment state
- `2`: SHORT ZigZag segment state

Contract rules:

- The active Omega1 action-label dataset is fixed as `3-class ZigZag action`.
- Canonical builder: `scripts/build_wave3_action_labels_20260531.py`.
- Canonical artifact directory:
  `tmp/causal_regen_20260516/zigzag_action_labels_20260531`.
- Canonical label files:
  - `zigzag_action_labels_2024.csv`
  - `zigzag_action_labels_2025.csv`
  - `zigzag_action_labels_2026.csv`
- Canonical hard label column: `zigzag_action`.
- Removed active label column: `wave3_action`. Do not silently alias it.
- Canonical risk-adjusted soft label columns:
  - `zigzag_soft_cash`
  - `zigzag_soft_long`
  - `zigzag_soft_short`
- Canonical audit:
  `tmp/causal_regen_20260516/zigzag_action_labels_20260531/zigzag_action_label_audit.json`.
- Active label method: ZigZag confirmed-pivot segments.
- Active ZigZag parameters: `zigzag_reversal_pct=0.010`,
  `min_wave_bars=8`, `transition_buffer=2`, `atr_multiplier=1.0`,
  `mae_penalty=1.25`, `softmax_temperature=1.75`,
  `min_risk_floor=0.0010`.
- Soft labels are target labels only. They are derived from future segment
  path return/MAE/MFE and must not be used as input features.
- Nearest-wave CASH expansion is disabled for ZigZag labels.
- Transition-buffer rows must remain CASH.
- Legacy Swing H/L wave3 and dense nearest-wave expansion are retired and must
  not be used in active Omega1 training.
- Omega1 teacher builders must read that explicit artifact and fail fast if it
  is missing.
- Legacy `tp_sl_action_score -> threshold -> 3-class` and TP/SL action labels
  are retired for Omega1. They may remain only in historical reports and must
  not be used as active labels.
- Any second-stage feature family trained on prior 2-action, binary
  long/short, or tradeable/no-trade action labels is stale for Omega1 active
  use. It must be retrained against `zigzag_action` and/or the explicit
  soft-label columns before promotion.
- Do not silently map an old 2-action/binary output into `zigzag_action`.
- `tp_sl_action_score`, path-edge scores, realized PnL, MFE/MAE, future path
  statistics, and any target-like wave construction columns must remain
  forbidden as teacher input features.
- Any future label replacement must update this section first; silent fallback
  from ZigZag labels to `tp_sl_action_score` is forbidden.

2-action-derived second-stage retrain notice:

- Retrain or keep research-only: old `ai_dir_*`, `ai_patch_*`,
  PatchTSMixer/PatchTST binary tradeable outputs, Alpha5/Alpha6
  direction/action heads, and any router/action meta features whose label
  contract was binary or 2-action.
- Retrain or keep excluded: old M7 direction/action heads such as
  `m7_trend_xgb_*`, `m7_mtl_*`, `m7_quant_*`, `m7_prob_*`, `m7_action`, and
  action/size/confidence/composite heads when their labels were 2-action or
  binary.
- Current ZigZag-action retrain artifacts to audit before promotion:
  - AI patch representation CatBoost head:
    `tmp/causal_regen_20260516/zigzag_ai_patchmix_catboost_20260531`.
  - M7 action HGB head:
    `tmp/causal_regen_20260516/zigzag_m7_action_hgb_20260531`.
- Exempt unless provenance says otherwise: risk/uncertainty/context features
  not trained on action labels, including Regime3 current context, Regime3 h6
  risk sidecar, TiDE risk outputs, and Chronos uncertainty outputs.

## Allowed Omega1 Teacher Inputs

Allowed model/context columns are explicit. Prefix sweeps such as `ai_*` or
`m7_*` are not allowed in Omega1 active/live candidates.

- AI/TSFM:
  - `ai_adverse_risk`
  - `ai_reward_risk`
  - `ai_vol_regime_pct`
  - `tide_vol_zscore`
  - `chronos_atr14_upside_band_ewm3`
  - `chronos_atr14_width_ewm6`
  - `chronos_atr14_width`
  - `chronos_atr14_large_move_score`
  - `chronos_realized_vol24_width`
  - `chronos_realized_vol24_large_move_score`
- M7:
  - `m7_q10`
  - `m7_q90`
  - `m7_qwidth`
  - `m7_quality_pred`
  - `m7_hold_pred`
- Regime3:
  - `regime3_current_sensitive_wide24_bull_prob`
  - `regime3_current_sensitive_wide24_bear_prob`
  - `regime3_current_sensitive_wide24_chop_prob`
  - `regime3_current_sensitive_wide24_confidence`
  - `regime3_current_sensitive_wide24_entropy`
  - `regime3_current_sensitive_wide24_margin`
  - `regime3_stability_h6_score`
  - `regime3_transition_h6_risk_prob`
  - `regime3_transition_h6_risk_pred`
  - `regime3_churn_h6_risk_score`
- Split-local current context:
  - `cvp_regime`
  - `regime_trending`

Research-only / excluded until a separate promotion test passes:

- PatchTSMixer / PatchTST direction outputs: `patchtst_*`, `pred_patchtst`, `conf_patchtst`
- DLinear outputs: `dlinear_*`
- broad direction outputs: `ai_dir_*`
- M7 legacy / diagnostic / raw-level outputs:
  - `m7_trend_xgb_*`, `m7_mtl_*`, `m7_quant_*`, `m7_prob_*`
  - `m7_action`, `m7_size`, `m7_confidence`, `m7_composite_score`
  - `m7_entry_*_price`, `m7_tp_price`, `m7_sl_price`
  - `m7_gmm_*`, `m7_iso_*`, `m7_vae_*`, `m7_hdb_*`, `m7_gate_block`

Regime3 sidecar joins must be exact timestamp joins. Tail-only missing rows may
be dropped. Missing values elsewhere are contract failures.

Regime3 current sidecar is allowed only as current market-structure context.
Allowed current sidecar prefix is `regime3_current_sensitive_wide24_*`.
Future-regime predictors such as `regime3_pred_*` remain excluded.

Chronos sidecar joins must also be exact timestamp joins. Chronos is allowed
only as uncertainty / large-move / downside-risk context, not as a hard
long/short direction owner.

## Current Implementation

Omega1 HGB teacher builder:

- `scripts/build_hgb_teacher_features_20260531.py`

The builder emits:

- `teacher_hgb_p_cash`
- `teacher_hgb_p_long`
- `teacher_hgb_p_short`
- `teacher_hgb_confidence`
- `teacher_hgb_side_edge`
- `teacher_hgb_uncertainty`
- `teacher_hgb_risk_veto_score`

These outputs are not promoted as direct action owners. Their intended use is
risk veto, size down, exit-risk context, threshold adjustment, and teacher-stack
diagnostics.

## Allowed Downstream Teacher Features

After generation, downstream Omega1 parent/risk/final-policy models may consume
the following teacher outputs:

- `teacher_hgb_p_cash`
- `teacher_hgb_p_long`
- `teacher_hgb_p_short`
- `teacher_hgb_confidence`
- `teacher_hgb_side_edge`
- `teacher_hgb_uncertainty`
- `teacher_hgb_risk_veto_score`

These are not allowed as inputs to the teacher builder itself.

## Omega1 Mamba Teacher Candidate

Native 72-step Mamba teacher candidate:

- Script: `scripts/train_omega1_mamba_teacher_20260531.py`
- Artifact: `tmp/causal_regen_20260516/omega1_mamba_teacher_current_chronos_seq72_20260531_e4`
- Sequence length: `72`
- Inputs: `27` Omega1 second-stage features + `90` base current-context features
- Input count: `117`
- Outputs:
  - `teacher_mamba_p_cash`
  - `teacher_mamba_p_long`
  - `teacher_mamba_p_short`
  - `teacher_mamba_confidence`
  - `teacher_mamba_side_edge`
  - `teacher_mamba_uncertainty`
  - `teacher_mamba_risk_veto_score`
- Label-probe metrics:
  - train bacc `0.7900`, train OVR AUC `0.9024`
  - 2025 internal validation bacc `0.3550`, OVR AUC `0.5494`
  - 2026 OOS bacc `0.4359`, OVR AUC `0.6264`

The Mamba teacher keeps `teacher_*` feedback inputs forbidden during teacher
generation. Its outputs may be tested only as downstream Omega1 parent/risk/final
policy inputs.

## Planning Notes

Omega1 should be evaluated as a layered teacher/risk stack:

```text
AI + M7 + Regime3 h6 context
  -> Omega1 teacher
  -> risk veto / size down / exit-risk context
  -> parent or final policy
```

Do not feed Omega1 teacher outputs back into AI, M7, Regime3, or router
generation without a new OOF/no-leak contract.
