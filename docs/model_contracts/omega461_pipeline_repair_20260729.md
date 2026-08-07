# Omega4.6.1 Pipeline Repair — 2026-07-29

## Outcome

The active account path remains disabled. The runtime now fails closed on unavailable entry
overlays, feature/artifact mismatches, corrupt position sizing state, and an incomplete promotion
manifest. No candidate in this repair was promoted to real execution.

## Implemented sequence

1. **Frozen current-state inventory**
   - `CURRENT_LIVE_MANIFEST.json` records runtime flags, artifact paths, SHA-256 values, sizing
     multipliers, source revision, and blockers without secrets.
2. **Single sizing contract**
   - All final Omega4.6.1 entry sizing passes through one cap operation.
   - The invariant is `notional = margin_fraction * leverage`; leverage is never multiplied into
     TP/SL a second time.
   - Entry state reconciliation rejects partial or internally inconsistent sizing state.
3. **Fail-closed entry overlays**
   - Overlay outcomes are `PASS`, `VETO`, or `UNAVAILABLE`.
   - Exceptions and insufficient history are `UNAVAILABLE`, never an implicit pass.
4. **Artifact and feature fail-fast contracts**
   - Runtime sidecars require validation-only selection, the exact q-tag, matching quality
     threshold, matching parent directory, and all three per-bar prediction artifacts.
   - Missing, duplicate, non-numeric, or non-finite sidecar features fail instead of receiving a
     synthetic zero.
5. **Validation-only risk selection**
   - The risk-sidecar selector has no OOS selection option or MDD-relaxation fallback.
   - It refuses to write a sidecar when the full validation replay misses trade-count or MDD
     constraints.
   - The artifact audit rejects reports without explicit constraint, fallback, and full-replay
     evidence.
6. **Fixed-seed research and selection statistics**
   - SOL entry direction/quality probabilities were averaged across seeds
     `17, 29, 43, 71, 101` before action thresholding.
   - The generated manifests explicitly state that exit heads and a live parent bundle are not
     ensembled, so these artifacts are not promotion eligible.
   - Reusable Sharpe, DSR, and PBO calculations are available in `core/selection_stats.py`.
   - A fixed-leverage deterministic sizing baseline is available for shadow comparison with the
     small-sample learned sidecar.
7. **Session and portfolio contracts**
   - `session_contract_v2` supplies timezone-aware NYSE cash-session, open-30-minute, weekend,
     and holiday features with DST handling and no legacy aliases.
   - The batch portfolio allocator applies fixed asset caps, same-direction caps, and a gross cap
     by proportional scaling. Results are independent of candidate input order.
   - Both remain research/shadow components; this repair does not silently activate them in live
     execution.
8. **Unified execution promotion gate**
   - An eligible manifest must contain all of the following:
     - artifact-integrity `promotion_pass=true`;
     - causal validation and OOS fresh-forward evidence;
     - `fresh_forward_bar_by_bar=true`;
     - `trade_ledgers_used_as_input=false`;
     - `saved_parent_exit_timestamps_used=false`;
     - `future_rows_used_for_entry=false`;
     - DSR at or above its recorded minimum and PBO at or below its recorded maximum.
   - `build_omega461_promotion_manifest_20260729.py` validates and hashes every evidence report
     before writing an eligible manifest.

## Current blockers

- The static current-live manifest is intentionally ineligible and the worktree is dirty.
- The existing SOL sidecars were selected with `validation_oos_guard`, so runtime lineage rejects
  them.
- The fixed-seed SOL artifacts ensemble entry direction/quality only. A causal live parent bundle
  with an ensembled exit head does not yet exist.
- No new candidate has produced passing bar-by-bar validation/OOS reports plus passing DSR/PBO and
  artifact-integrity evidence.
- Session features and the batch portfolio allocator have not been trained/evaluated in a fresh
  candidate and are not wired into real execution.

## Required promotion order

1. Build a candidate with either a complete parent/exit ensemble or the deterministic sizing
   baseline; do not attach a single-seed learned risk sidecar to the entry-only ensemble.
2. Run causal bar-by-bar validation, select all thresholds on validation only, and freeze them.
3. Compute DSR/PBO for the fixed candidate family and apply the recorded thresholds.
4. Run the frozen candidate once on OOS without reselection.
5. Run `audit_omega_artifact_integrity_20260630.py` and require exit status 0 plus
   `promotion_pass=true`.
6. Build the promotion manifest from the four evidence reports.
7. Only after the runtime accepts that manifest may account execution be considered separately.

## Verification note

The focused contract tests run with the `quant_ai` Python interpreter and standard-library
`unittest`. The environment does not currently contain `pytest`, so the repository-wide pytest
suite was not run as part of this repair.
