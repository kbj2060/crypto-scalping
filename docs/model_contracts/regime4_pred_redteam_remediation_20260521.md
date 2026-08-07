# Regime4 Pred Red-Team Remediation

Date: 2026-05-21

## State24 Artifact Origin

`clean_regime4_state24_sticky090_v2_20260517` was generated on 2026-05-17 KST. The model file timestamp is `2026-05-17 23:47:50 +0900`, and the 2024/2025/2026 sidecars plus report were written by `2026-05-17 23:48:04 +0900`.

It is a 2024-fit 24-state HMM with sticky `0.90`; it emits the same legacy CSV prefix `clean_regime4_2024_unsup_v1_*` in the original artifact files, so downstream fixed DSAC inventory renames those outputs to `clean_regime4_state24_sticky090_v2_*`.

## Verdict

`regime4_pred_tft_h12_nomdjd_all74_20260517` is the canonical Regime4 future predictor for DSAC soft-feature use and preprocessing. The older `regime4_pred_tft_vsn_h12_official_20260517` artifact is deprecated for active paths.

## Findings

1. The old h12 official artifact depended on `pred_mdjd`/`conf_mdjd`.
2. The 2026 rebuilt frame did not contain those fields.
3. The transform path previously median-filled missing TFT inputs, which made the old artifact look runnable while silently changing its input distribution.
4. DSAC feature inventory merged state24 clean-regime columns under the same `clean_regime4_2024_unsup_v1_*` prefix as the raw-state12 lineage, creating provenance ambiguity.
5. Validation metrics are selection evidence only. Final promotion still requires downstream DSAC/backtest ablation on a frozen split not used for selecting the Regime4 predictor.

## Applied Fixes

1. `regime4_official_v1_20260517_contract.md` now names the no-mdjd all74 artifact as canonical and marks the old mdjd artifact as deprecated.
2. `transform_regime4_official_sidecars_20260517.py` now fails closed when TFT feature columns are missing. Median fallback requires an explicit command-line override and one allowlisted feature argument per missing column.
3. `build_dsac_feature_inventory_20260521.py` now writes to `dsac_feature_inventory_regime_fixed_20260521` by default, drops ambiguous base `clean_regime4_2024_unsup_v1_*` columns, and replaces them with renamed state24 clean-regime features under `clean_regime4_state24_sticky090_v2_*`.
4. DSAC family PCA and variant spec builders now default to the fixed inventory/spec directories and keep `clean_regime4_state24` and `regime4_pred` as separate families.

## Active DSAC Naming Contract

For DSAC feature inventory and future Router/DSAC experiments, `clean_regime4_2024_unsup_v1_*` is an ambiguous legacy export prefix and must not appear in active feature specs.

Active DSAC regime features must use:

```text
clean_regime4_state24_sticky090_v2_*
regime4_pred_*
```

The fixed inventory builder enforces this by dropping base `clean_regime4_2024_unsup_v1_*` columns and re-merging the state24 sidecar under `clean_regime4_state24_sticky090_v2_*`. Historical Alpha5/Alpha5.1/Alpha5.2 reports may still contain the legacy prefix because those experiments were run before the provenance rename.

`clean_regime4_2024_unsup_v1_*` is allowed only for historical reproduction or for reading the original state24/raw-state artifact CSVs before the fixed DSAC rename step. It is a contract violation in active DSAC/Router feature specs.

## Fixed Paths

```text
/home/llewyn/crypto-scalping/tmp/causal_regen_20260516/dsac_feature_inventory_regime_fixed_20260521/
/home/llewyn/crypto-scalping/tmp/causal_regen_20260516/dsac_feature_variant_specs_regime_fixed_20260521/
```

Current fixed spec verification:

```text
legacy clean_regime4_2024_unsup_v1_* columns: 0
active state24 clean_regime4_state24_sticky090_v2_* columns: present
```

The currently running DSAC feature batch still uses the older inventory path and was not modified in-place.
