# Lifecycle V1 Canonical Accounting Audit 2026

Verdict: `audit_artifact_only_not_promoted`

## Purpose

Freeze Lifecycle V1 as the best current research substrate and audit its accounting package without adding a new alpha layer.

## Key Results

- Lifecycle V1 OOS PnL 1x: `207.236888%`
- Lifecycle V1 OOS MDD 1x: `-18.016318%`
- Trades/day: `6.187500`
- Fixed-ledger 3x PnL: `52.844616%`
- Path-changing 3x PnL: `-8.649827%`
- Preservation audit passed: `True`
- Giveback carry audit passed: `True`

## Cost Summary

| View | 1x PnL | 2x PnL | 3x PnL | 3x Trades |
|---|---:|---:|---:|---:|
| Path-changing | `207.236888%` | `133.048111%` | `-8.649827%` | `179` |
| Fixed ledger | `207.236888%` | `116.731651%` | `52.844616%` | `363` |

## Cost Separation

This package reports two different cost views:

- `cost_path_changing_*`: rebuilds the clean-base trade path under each fee/slippage multiplier.
- `cost_fixed_ledger_*`: keeps the 1x Lifecycle V1 ledger fixed and changes only fee/slippage.

## Promotion Decision

Lifecycle V1 remains a research candidate, not a promoted production model. It improves PnL over the clean base but misses the clean-base MDD gate, fails path-changing 3x survival, and still lacks realistic funding/impact/partial-fill replay.

## Artifacts

- Report: `/home/llewyn/crypto-scalping/data/ensemble/reports/lifecycle_v1_canonical_accounting_audit_2026.json`
- Ledger: `/home/llewyn/crypto-scalping/data/ensemble/reports/lifecycle_v1_canonical_accounting_audit_2026_ledger.csv`
- Cost CSV: `/home/llewyn/crypto-scalping/data/ensemble/reports/lifecycle_v1_canonical_accounting_audit_2026_fixed_ledger_cost.csv`
