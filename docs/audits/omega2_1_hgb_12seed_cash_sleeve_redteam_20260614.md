# Omega2.1 HGB 12-Seed Cash Sleeve Red-Team Audit - 2026-06-14

## Verdict

`omega2_1_label_atr1_h24_hgb_12seed_ensemble_thr055` is deprecated as historical reference only.

Status:

`deprecated_historical_reference_only_accounting_invalid_true_leverage`

## What Passed

- Manifest and frozen bundle feature lists match.
- Frozen bundle model id matches the expected model id.
- Direct forbidden feature contract passes for the 42 stored input columns.
- Runtime adapter is fail-fast on missing/non-finite/forbidden columns.

## What Failed

The original replay treated `notional_exposure` as the effective PnL/fee/MDD exposure while also storing `leverage = 2.0`.

Current Omega accounting requires:

`effective_exposure = notional * leverage`

Therefore the headline result is not promotion-safe.

## Metrics

Legacy reported OOS:

- PnL: `+102.611483%`
- MDD: `-8.108171%`
- WR: `60.975610%`
- trades: `41`
- fallback entries: `23`
- primary takeovers: `12`

Corrected true-leverage OOS:

- PnL: `+33.877901%`
- MDD: `-23.976364%`
- WR: `43.410853%`
- trades: `129`

## Recommendation

- Do not live-promote this artifact.
- Do not use the legacy `+102.61%` result as promotion evidence.
- Rebuild and re-evaluate under true-leverage accounting if the architecture is reused.

## Artifact

- Report: `tmp/causal_regen_20260516/omega2_1_hgb_12seed_redteam_audit_20260614/report.json`
