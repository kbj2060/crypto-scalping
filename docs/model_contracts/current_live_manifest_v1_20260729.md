# Current Live Manifest v1 — 2026-07-29

`CURRENT_LIVE_MANIFEST.json` is the sole machine-readable snapshot of the active
Omega4.6.1 runtime configuration while the promotion pipeline is being repaired.

## Scope

- Records non-secret execution flags and active model identifiers.
- Records the exact paths, sizes, and SHA256 hashes of parent bundles, risk
  sidecars, sidecar reports, and per-asset regime models.
- Records sidecar selection scope and exact precomputed-prediction lineage when
  the report provides it.
- Records Git commit and dirty-worktree counts without storing the file list.

The manifest is a snapshot, not a promotion certificate. Its
`promotion_eligible` field must remain `false` until the unified promotion gate
is implemented and passes. Missing artifacts, non-validation-only selection,
or a dirty worktree must never be silently corrected.

## Regeneration

Run `scripts/build_current_live_manifest_20260729.py` in the `quant_ai`
environment and compare its JSON output with `CURRENT_LIVE_MANIFEST.json`.
Do not include `.env` values other than the explicit boolean safety flags.

## Safety

This contract does not enable trading. `BINANCE_ACCOUNT_ENABLED` and real
execution flags remain independent hard safety gates. Until the P0 runtime
contracts pass, account execution must remain disabled.
