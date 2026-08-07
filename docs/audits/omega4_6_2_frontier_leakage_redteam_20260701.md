# Omega 4.6.2 Frontier Leakage Red-Team Audit - 2026-07-01

- Verdict: `FRONTIER_LEAKAGE_RUNTIME_PASS`
- Direct future-data leak found: `False`
- Data contamination found: `False`
- OOS selection contamination blockers: `[]`
- Full live pass: `True`

## Model Verdicts

| Model | Direct Leak | Split Clean | Entry Feature Parity | OOS Selection Risk | Verdict |
| --- | --- | --- | --- | --- | --- |
| `omega4_6_2_v5_roll8_side_specific_two_stage_exposure_validation_only_20260701` | `False` | `True` | `True` | `readout_after_selection` | `NO_DIRECT_LEAK_FOUND` |

## Critical Findings

- No direct future-data leakage was found in the audited ledgers/features.

## OOS/Fresh-Holdout Findings

- `omega4_6_2_v5_roll8_side_specific_two_stage_exposure_validation_only_20260701`: OOS selection mode `readout_after_selection`, fresh holdout declared `False`.

## Entry Feature Causality

- For models using `volume` and `cvp_vah_val_width`, the audit compared ledger feature values against the source market CSV at each active trade's `entry_timestamp`.
- The corrected check uses `entry_timestamp`, because rolled segment `entry_i` values are synthetic and can exceed the source CSV row count.
- Upstream CVP feature causality audit: `CVP_FEATURE_CAUSALITY_PASS` at `/home/llewyn/crypto-scalping/tmp/causal_regen_20260516/cvp_feature_causality_20260701/cvp_feature_causality_20260701.json`.

## Artifacts

- JSON: `/home/llewyn/crypto-scalping/tmp/causal_regen_20260516/omega4_6_2_frontier_leakage_redteam_20260701/frontier_leakage_redteam_20260701.json`
