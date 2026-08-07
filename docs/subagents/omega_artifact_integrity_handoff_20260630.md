# Omega Artifact Integrity Handoff

Date: 2026-06-30 KST

This handoff records the fail-fast rule all subagents must apply to future Omega/Omega4.x model upgrades, red-team reviews, baseline changes, and live promotion checks.

## Required Gate

- Canonical policy: `docs/model_contracts/omega_artifact_integrity_policy_20260630.md`
- Audit script: `scripts/audit_omega_artifact_integrity_20260630.py`
- Promotion requires the audit to return exit status 0 and `promotion_pass=true`.
- If the audit fails, keep the existing baseline or treat the candidate as diagnostic-only.

## Artifact Requirements

- Parent model runs must save exact-threshold per-bar prediction files for each promoted quality threshold:
  - `train_predictions_qXXX.csv`
  - `validation_predictions_qXXX.csv`
  - `oos_predictions_qXXX.csv`
- `qXXX` is `round(quality_threshold * 100)` zero-padded, for example `q055` for `0.55`.
- Risk sidecars that consume parent outputs must record:
  - `risk_model.precomputed_prediction_dir`
  - `risk_model.precomputed_prediction_tag`
- Saved trade ledgers, candidate-event ledgers, and historical comparison ledgers are diagnostic-only. They cannot replace per-bar parent prediction artifacts for promotion.

## Audit Command

```bash
cd /home/llewyn/crypto-scalping
PYTHONPYCACHEPREFIX=/tmp/quant_ai_pycache_20260630 /home/llewyn/miniconda3/envs/quant_ai/bin/python scripts/audit_omega_artifact_integrity_20260630.py --report <candidate_report.json>
```

## Current Blocked Example

The current Omega4.5 `v5_guard18p0` candidate was intentionally marked blocked until the missing artifact contract is regenerated:

- Candidate report: `tmp/causal_regen_20260516/omega4_5_baseline_v5_guard18p0_20260630/report.json`
- Audit JSON: `tmp/causal_regen_20260516/omega4_5_baseline_v5_guard18p0_20260630/omega_artifact_integrity_audit_20260630.json`
- Audit Markdown: `tmp/causal_regen_20260516/omega4_5_baseline_v5_guard18p0_20260630/omega_artifact_integrity_audit_20260630.md`

Known blockers from that audit include missing exact-threshold parent predictions for several components and missing `risk_model.precomputed_prediction_dir/tag` in risk sidecar reports.
