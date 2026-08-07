# SOL dual-component H24-wide final report (2026-07-29)

> **Seed-stability update:** the candidate failed a five-seed frozen-policy
> retraining test. It must not be promoted to live or used as the new baseline.
> See `docs/sol_dual_h24wide_seed_stability_20260729.md`.

## Outcome

The originally selected candidate is a genuine single-slot, bar-by-bar dual ensemble:

- bull: H24-wide parent q0.55, margin scale 0.25
- bear: H24-wide parent q0.55, margin scale 0.50
- chop: ZIGZAG parent q0.60, margin scale 1.00
- each component keeps its own parent, ATR TP/SL, Exit Head, and risk sidecar
- validation selection only; canonical OOS was evaluated once after freezing

| Window | Candidate PnL | Candidate MDD | Baseline PnL | Baseline MDD |
| --- | ---: | ---: | ---: | ---: |
| Validation 2025-09-01..12-31 | +25.08% | -7.49% | +23.45% | -7.69% |
| OOS 2026-01-01..03-31 | +21.62% | -9.88% | +7.66% | -12.52% |

The candidate improves both PnL and MDD in both required windows. Artifact integrity audit returned `promotion_pass=true` for both exact-threshold component chains.

## Label continuity

H24 conservative raw labels flipped substantially more often than ZIGZAG. Split-local Potts smoothing with mismatch cost 1 and switch penalty 12 was retained, matching the previously approved wide-H24 chart.

| Split | ZIG direct flip interval | Raw H24 | H24-wide |
| --- | ---: | ---: | ---: |
| Train | 3.47 h | 1.31 h | 16.13 h |
| Validation | 4.04 h | 1.50 h | 14.82 h |
| OOS | 4.49 h | 1.52 h | 16.81 h |

The penalty grid (0, 3, 6, 9, 12, 18, 24) was recorded in the label report. Labels were generated independently inside train, validation, and OOS boundaries; no cross-split future row was used.

## Architecture tests

The tested policy families included both fixed priority orders, conflict-to-cash, unanimity-only, all eight bull/bear/chop hard expert assignments, duration gating, global margin scaling, and regime-specific margin scaling. The successful structure-aware router follows the same core principle as AME-TS: route heterogeneous experts using an interpretable temporal regime prior. MM-DREX independently supports separating market-state perception from specialist execution.

- AME-TS: https://huggingface.co/papers/2605.25166
- MM-DREX: https://huggingface.co/papers/2509.05080

## Evaluation and integrity contract

- `fresh_forward_bar_by_bar=true`
- `trade_ledgers_used_as_input=false`
- `saved_parent_exit_timestamps_used=false`
- `future_rows_used_for_entry=false`
- `oos_used_for_selection=false`
- one shared position slot
- `notional = margin_fraction * leverage`
- leverage cap 5.0; notional cap 1.8

## Diagnostics and caveats

Monthly OOS PnL was +13.71% in January, +11.81% in February, and -4.34% in March. March weakness was not used for tuning because it is OOS. The candidate is stronger than the referenced fresh-retrain baseline, but the validation PnL margin is only +1.63 percentage points and should be monitored in fresh forward data before live replacement.

Canonical artifacts:

- Label report: `tmp/causal_regen_20260516/sol_dual_zig075_h24wide_splitlocal_20260729/report.json`
- Router report: `tmp/causal_regen_20260516/sol_dual_structure_router_sidecar_q060_q055_20260729/report.json`
- Integrity audit: `tmp/causal_regen_20260516/sol_dual_structure_router_sidecar_q060_q055_20260729/integrity_audit/omega_artifact_integrity_audit_20260630.json`
