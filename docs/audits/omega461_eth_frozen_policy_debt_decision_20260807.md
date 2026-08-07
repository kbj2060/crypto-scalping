# ETH Omega4.6.1 frozen-artifact policy-debt decision (2026-08-07)

## What was found

End-of-session integrity sweep (2026-08-07) ran `scripts/audit_omega_artifact_integrity_20260630.py`
against the LIVE ETH stack's replay report (`tmp/causal_regen_20260516/omega4_6_1_extended_oos_20260706/report.json`,
components h48qual + zig075). Result: `promotion_pass=false`, with six failures per component:

- `risk_constraint_pass_declared`, `risk_fallback_not_used`, `risk_machine_readable_constraints_present`,
  `risk_full_replay_metrics_meet_constraints` -- machine-readable constraint declarations the sidecar
  trainer only began writing after the 2026-06-30/07-30 policy iterations.
- `risk_dataset_lineage_present`, `parent_dataset_lineage_present` -- the `dataset_lineage` block
  introduced by P0-2 (2026-07-30, `docs/pipeline_integrity_and_research_redesign_20260730.md`).

**All core integrity checks PASS for both components**: parent bundles exist, sidecars use exact-
threshold precomputed parent predictions (q050/q075), and train/validation/oos prediction timestamps
match the runtime frame exactly.

## Decision: documented exception (grandfathered), NOT backfill, NOT retrain

1. **No lineage backfill.** The audit's own docstring states the lineage gate "deliberately fails
   every report.json written before this gate existed." The 07-06 frozen ETH artifacts are the
   canonical example of WHY: `training_features_2026_rebuilt.csv` changed in place (upstream Binance
   metrics zips retroactively revised) and the exact bytes the frozen artifacts consumed are
   **unknowable** (see `project-omega461-baseline-drift-bisection-20260730`). Backfilling
   `dataset_lineage` with TODAY's file hashes would assert a provenance that is provably not the
   training provenance -- a fabricated record. Same logic applies to `constraint_pass` /
   `trade_floor` / `mdd_floor`: the training runs never declared or evaluated those constraints.

2. **No retrain to satisfy the gate.** ETH parent retraining is known non-reproducible
   (21+ exit-logic rounds and the drift bisection both confirmed retrained artifacts do not
   reproduce the frozen baseline). Retraining to regenerate policy-compliant reports would replace
   the validated live model with a different, unvalidated one -- strictly worse for integrity.

3. **Grandfather clause.** The frozen ETH stack was promoted live on 2026-07-06 under the gate
   version in force at that time (`promotion_pass=true` then; see
   `docs/model_contracts/omega4_6_1_live_path_parity_and_lookahead_audit_20260706.md` era records).
   Its live status is retained. This mirrors the existing SOL precedent
   (`allowed_selection_scopes` exception in `omega4_6_1_live.py`, 2026-07-20) and the BTC
   asset-specific-audit precedent (2026-07-13, reaffirmed 2026-08-07).

## Binding rule going forward

- The exception covers ONLY the frozen 07-06 ETH artifacts currently live. **Any future ETH
  candidate (retrain, replacement, or upgrade) must pass the full current
  `audit_omega_artifact_integrity_20260630.py` gate, including dataset_lineage** (its datasets
  registered via `scripts/dataset_snapshot.py` at build time).
- The same applies to BTC: the 2026-08-07 swingtransition promotion passed the BTC asset-specific
  gate (its own grandfathered convention); the NEXT BTC candidate should be built with
  lineage-registered datasets so the generic gate passes without exception.
- This document is the audit-trail record of that decision; the audit outputs live at
  `tmp/causal_regen_20260516/omega4_6_1_extended_oos_20260706/omega_artifact_integrity_audit_20260630.{json,md}`.
