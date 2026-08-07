# Secondary Feature Generation Contract - 2026-05-30

## Purpose

This note records how the current second-stage features are generated and which
dependencies are forbidden. It exists to prevent upstream feature builders from
accidentally consuming downstream artifacts, especially the previous mistake of
trying to feed `teacher_*` features into AI feature generation.

Active/live paths must fail fast on contract mismatch. Do not add silent alias,
fallback prefix, or compatibility repair logic to active candidates.

## Dependency Order

```text
clean base features
  ├─> AI / TSFM forecasts: pred_patchtst, conf_patchtst, ai_*, patchtst_*, tide_*, timesnet_*, dlinear_*
  ├─> M7 ensemble outputs: m7_*
  ├─> current/future regime surfaces: clean_regime4_state24_sticky090_v2_*, regime4_pred_*
  └─> selected router inputs

AI / TSFM + M7
  └─> teacher_* side/meta features

selected router inputs
  └─> a5dir_* router probabilities

final policy/risk models
  └─> may consume allowed AI/M7/teacher/regime/a5dir features by explicit feature contract
```

The direction is one-way. A downstream family must never be fed back into an
upstream generator under the same prefix/version.

## Family Contracts

| Family | Prefix / columns | Generation path | Upstream inputs | Downstream use | Forbidden as input to |
|---|---|---|---|---|---|
| AI / TSFM forecasts | `pred_patchtst`, `conf_patchtst`, `ai_*`, `patchtst_*`, `tide_*`, `timesnet_*`, `dlinear_*` | `ensemble/ensemble_router.py`, `pipeline/augment_alternative_features.py`, `pipeline/build_unified_rl_dataset.py`, `features/model_adapters.py`; see `docs/audits/ai_direction_feature_retrain_20260528.md` | Clean causal base features and model-specific frozen schemas | Teacher generation, final policy context, feature analysis | `teacher_*`, `a5dir_*`, M7 outputs, regime outputs, PCA experiment outputs, labels/targets unless a new versioned artifact explicitly audits that dependency |
| M7 ensemble outputs | `m7_*` | `scripts/train_all_ensemble.py`, `ensemble/seven_model_ensemble.py`, `pipeline/augment_m7_dataset.py`, `pipeline/build_unified_rl_dataset.py` | Clean 2024 training split and M7 model stack inputs, scored into 2025/2026 by exact timestamp merge | Teacher generation, final policy context, router/final experiments if explicitly listed | `teacher_*`, `a5dir_*`, AI outputs, regime outputs, PCA experiment outputs, labels/targets unless a new versioned artifact explicitly audits that dependency |
| Teacher side/meta features | Current Alpha8: `teacher_long_edge`, `teacher_short_edge`, `teacher_side_margin`, `teacher_side_disagreement`, `teacher_quantile_skew`, `teacher_uncertainty`, `teacher_tail_warning` | `pipeline/teacher_meta_side_features.py`; older/general meta path `pipeline/certified_teacher_meta_features.py`; certified builders `scripts/build_certified_teacher_features_2025.py`, `scripts/build_certified_teacher_features_2026.py` | AI/TSFM outputs plus M7 outputs at the same timestamp | Final policy/risk models only | AI/TSFM generation, M7 generation, regime generation, a5dir/router generation |
| Current regime surface | `clean_regime4_state24_sticky090_v2_*` | Source HMM family from `scripts/retrain_clean_regime4_hmm_raw_state12_20260517.py`; active rename/merge path in `scripts/build_dsac_feature_inventory_20260521.py`; active prefix recorded in `docs/feature_contract_manifest.json` | Raw current-row regime/state features under the artifact contract | Final policy/risk context; selected feature tests; future-regime labels/context where explicitly audited | `regime4_pred_*`, `a5dir_*`, `teacher_*`, AI outputs, M7 outputs, PCA experiment outputs unless a new versioned artifact explicitly audits that dependency |
| Future regime surface | `regime4_pred_*` | `scripts/build_regime4_pred_tft_clean_target_20260517.py`, `scripts/build_regime4_pred_tft_vsn_select_20260517.py`, `scripts/transform_regime4_official_sidecars_20260517.py`; artifact `data/ensemble/supervised/regime4_pred_tft_h12_nomdjd_all74_20260517`; regenerated after clean funding fix | Raw features plus current Regime4 sidecar under h12/all74 contract; labels are future current-regime class at horizon | Final policy/risk context; selected feature tests | current-regime HMM generation, AI/TSFM generation, M7 generation, teacher generation, a5dir/router generation |
| Router probabilities | `a5dir_*` | `scripts/build_alpha5_a5dir_2024_train_2025_score_20260521.py`, `scripts/alpha5_router_v5_train_20260520.py`, `scripts/alpha5_direction_router_score_rl_csv_20260519.py`; clean run `tmp/causal_regen_20260516/alpha5_a5dir_2024_train_2025_score_20260529_cleanfunding_stable48` | Selected clean candidate features for router training/scoring plus explicit router artifacts | Final policy/risk context or router diagnostics | AI/TSFM, M7, teacher, current/future regime generation, Alpha5 label/router artifact generation |
| PCA / compressed variants | `*_pca*`, experiment-specific PCA columns | `scripts/add_dsac_family_pca_20260521.py`, feature-screening scripts | Clean candidate feature family used for the experiment | Downstream DSAC/Alpha6/Alpha7 experiments only | AI/TSFM, M7, teacher, regime, a5dir/router generation |

## Teacher Feature Formula

Current side-teacher features are deterministic combinations from
`pipeline/teacher_meta_side_features.py`.

Inputs:

- AI side estimates:
  - `ai_dir_p_up`, fallback from positive `pred_patchtst`
  - `ai_dir_p_down`, fallback from negative `pred_patchtst`
  - `conf_patchtst`
- M7 side/risk estimates:
  - `m7_trend_xgb_up`, fallback `m7_prob_up`
  - `m7_trend_xgb_dn`, fallback `m7_prob_dn`
  - `m7_confidence`
  - `m7_q10`, `m7_q50`, `m7_q90`
  - `m7_tail_risk`
- Optional AI risk:
  - `ai_adverse_risk`

Outputs:

```text
teacher_long_edge        = confidence-weighted AI/M7 long probability
teacher_short_edge       = confidence-weighted AI/M7 short probability
teacher_side_margin      = teacher_long_edge - teacher_short_edge
teacher_side_disagreement= AI/M7 sign disagreement weighted by combined confidence
teacher_quantile_skew    = m7_q90 + m7_q10 - 2 * m7_q50
teacher_uncertainty      = clip(abs(m7_q90 - m7_q10) / 0.02, 0, 3)
teacher_tail_warning     = clip(ai_adverse_risk + m7_tail_risk + 0.25 * teacher_uncertainty, 0, 3)
```

Because `teacher_*` depends on AI and M7 outputs, it is a downstream meta
feature. It is valid as final model input only after the AI/M7 artifacts have
already been trained and score-only generated for the target year.

## Certified Teacher Builders

- `scripts/build_certified_teacher_features_2025.py`
  - Base 2025: `data/splits/year_oos/training_features_2025.csv`
  - AI 2025: `data/tmp/unified_build_ckpt/03_after_ai.csv`
  - M7 2025: `data/splits/year_oos/rl_training_2025_m7.csv`
  - Fits clean regime on 2024 only, then scores 2025.
  - Writes `data/ensemble/supervised/certified_teacher_regime_moe_v1/features_2025.csv`.
- `scripts/build_certified_teacher_features_2026.py`
  - Base 2026: `data/splits/year_oos/training_features_2026_rebuilt.csv`
  - AI 2026: `data/tmp/unified_build_ckpt_2026/03_after_ai.csv`
  - M7 2026: `data/splits/year_oos/rl_training_2026_m7_supervised_redesign_clean.csv`
  - Loads frozen 2024 regime artifact, then scores 2026.
  - Writes `data/ensemble/supervised/certified_teacher_regime_moe_v1/features_2026.csv`.

Both certified paths use exact timestamp joins. Active paths must not use
`merge_asof`, backfill, or future shift to attach these features.

## Hard Rules

1. `teacher_*` is never an input to AI/TSFM, M7, regime, or a5dir/router
   generation. It is downstream of AI plus M7.
2. AI/TSFM outputs, M7 outputs, and current/future regime outputs should not be
   cross-fed into each other under existing prefixes. If such coupling is
   intentionally tested, create a new versioned artifact and no-leak audit.
3. `a5dir_*` is a downstream router output. It is never an input to AI/TSFM,
   M7, teacher, or regime generation.
4. Legacy regime prefixes are historical/reference-only and blocked for active
   inputs:
   - `clean_regime_2024_unsup_v4_*`
   - `clean_regime4_2024_unsup_v1_*`
5. Active regime inputs must use the explicit active surfaces:
   - `clean_regime4_state24_sticky090_v2_*`
   - `regime4_pred_*`
6. Raw M7 price outputs remain direct-input exclusions unless a new audited
   experiment explicitly promotes them:
   - `m7_entry_long_price`
   - `m7_entry_short_price`
   - `m7_tp_price`
   - `m7_sl_price`
7. If a required upstream feature family is missing during active materialization,
   fail fast. Do not silently fill missing active features with `0`, rename
   prefixes, or use fallback families.
8. If a generator intentionally consumes a family that was previously downstream,
   it must use a new versioned prefix/artifact and a new no-leak audit.

## Clean Funding Provenance

Secondary features that consumed funding-derived columns are clean only when
their manifest or input path points to the 2026-05-29 clean funding remediation
chain:

- `docs/audits/funding_clean_retrain_rescore_20260529.md`
- `docs/audits/funding_feature_redteam_20260529.md`
- `data/ensemble/reports/m7_teacher_live_provenance_20260527_audit.json`

Older cached DSAC, Alpha6, Alpha7, Alpha8, M7, teacher, or router inputs remain
research-only unless their manifest proves clean funding retraining/rescoring.
