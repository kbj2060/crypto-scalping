# Omega4.6.2 TCN Sequence Entry Gate Short Variants - 2026-07-03

## Scope

- Parent: `Omega4.6.2 source_v5`
- Gate artifact: `tcn_seq_gate_L24_flat.pt`
- Runtime change only: short-only side filter and threshold variants
- Validation: `2025-09-01 00:00:00` to `2026-01-01 00:00:00`
- OOS: `2026-01-01 00:00:00` to `2026-04-01 00:00:00`
- Replay: fresh-forward, 5m bar-by-bar, no ledger input

## Best Candidate

`tcn_seq_gate_L24_flat_short_thrm0p008412_short`

- Threshold: `-0.008412085473537445`
- Side filter: short-only
- Validation compound PnL: `+2.8570%`
- Validation compound MDD: `-17.6130%`
- Validation trades: `112`
- Validation WR: `40.18%`
- OOS compound PnL: `+58.1871%`
- OOS compound MDD: `-12.8849%`
- OOS trades: `57`
- OOS WR: `50.88%`

## Variant Ranking

| Variant | Validation PnL | Validation MDD | Validation Trades | OOS PnL | OOS MDD | OOS Trades |
|---|---:|---:|---:|---:|---:|---:|
| `thr=-0.008412` short-only | `+2.86%` | `-17.61%` | `112` | `+58.19%` | `-12.88%` | `57` |
| `thr=-0.013343` short-only | `-6.73%` | `-24.12%` | `122` | `+57.72%` | `-12.03%` | `59` |
| `thr=-0.004099` short-only | `-6.97%` | `-28.65%` | `105` | `+30.53%` | `-20.06%` | `56` |

## Integrity

All replay integrity counters are zero for every variant:

- `ledger_replay_trace_count = 0`
- `non_live_native_trace_count = 0`
- `non_minus_one_policy_row_count = 0`

## Interpretation

The previous full TCN run showed `L24 flat` was the best general sequence gate, but its long trades were the main validation drag. Applying a short-only runtime side filter turns the same artifact into a cleaner candidate:

- Validation moves from negative to positive.
- OOS remains strongly positive.
- Validation and OOS MDD both stay inside `-20%`.

This supports splitting the stack by side: keep the short specialist path, and do not re-enable long entries without a separate long-specialist gate.

## Artifacts

- Variant report: `tmp/causal_regen_20260516/omega462_live_native_tcn_sequence_entry_gate_L24_flat_short_variants_20260703/report.json`
- Source TCN report: `tmp/causal_regen_20260516/omega462_live_native_tcn_sequence_entry_gate_20260703/report.json`
- Eval script: `scripts/eval_omega462_sequence_gate_variants_20260703.py`
- Train/eval script: `scripts/train_eval_omega462_live_native_sequence_entry_gate_20260703.py`
