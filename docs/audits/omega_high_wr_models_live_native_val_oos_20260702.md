# Omega High-WR Candidates Live-Native Validation/OOS Audit - 2026-07-02

## Scope

Request: retest the previously identified high win-rate Omega candidates with live-native validation/OOS.

Fresh-forward contract used here:
- validation: `2025-09-01 00:00:00` to `2026-01-01 00:00:00` exclusive
- OOS: `2026-01-01 00:00:00` to `2026-04-01 00:00:00` exclusive
- 5m bar-by-bar replay
- no saved trade ledger as input
- no saved parent entry/exit timestamps
- no future rows for entry decisions

## Runner

Runner:
`scripts/run_omega462_hf_policy_bar_forward_val_oos_20260702.py`

Live-native parent variants:
- `source_v5`: Omega4.6.2 source parent live adapter with V5 exposure/loss-governor state.
- `cap220_no_v5`: test-only subclass in the runner that uses the cap220 live parent path but disables V5 exposure, for cap200/cap220-family candidate checks.

Additional validation feature input for roll8:
`tmp/causal_regen_20260516/live_native_inputs/training_features_2025_with_m7_prob_up_20260702.csv`

This file merges only current-row predictive `m7_prob_up` by timestamp into the base 2025 feature frame. It is not a trade ledger.

## Results

| Candidate | Parent Variant | Validation PnL | Validation MDD | Val Trades | Val WR | OOS PnL | OOS MDD | OOS Trades | OOS WR | Integrity |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `omega4_6_2_loss_cluster_governor_v5_fine_exposure_20260701` | `source_v5` | -32.05% | -32.26% | 153 | 32.68% | +53.36% | -19.84% | 75 | 42.67% | PASS |
| `omega4_6_borrowed_upgrade_notional_cap200_20260630` | `cap220_no_v5` | -17.78% | -20.21% | 153 | 32.68% | +22.53% | -12.14% | 75 | 42.67% | PASS |
| `omega4_6_2_cap220_paper_optstop_exit_overlay_20260701` | `cap220_no_v5` | -17.78% | -20.21% | 153 | 32.68% | +22.53% | -12.14% | 75 | 42.67% | PASS |
| `omega4_6_2_v5_roll8_side_specific_two_stage_veto_20260701` | `source_v5` | -34.38% | -39.90% | 277 | 44.40% | +28.70% | -11.52% | 125 | 50.40% | PASS |

Integrity means:
- `ledger_replay_trace_count = 0`
- `non_live_native_trace_count = 0`
- `non_minus_one_policy_row_count = 0`

## Notes

- All four Omega4.x candidates fail promotion on validation PnL under the live-native contract.
- The earlier high-WR reports were sparse ledger/post-hoc lifecycle diagnostics. In live-native replay, the parent emits many more current-row opportunities, so the trade distribution changes materially.
- `cap220_paper_optstop` equals `borrowed_cap200` in this run because all closed trades hit TP/SL before the 120h hard stop or 72h/96h opt-stop rules.
- `roll8_m7_veto` kept max hold at 8h, but the m7 veto fired only once in validation and once in OOS under this corrected split/input frame.
- `omega1_2_1_short_cap_seed_stability_20260612` was not run as live-native. Its script builds labels and simulates from historical `runner._build()` decision/state payloads; no equivalent short-cap live adapter artifact was found. The current Omega1 live adapter is for `omega3_aggressive_compensated_scale200_cap090_20260618`, not this seed-stability model.

## Artifacts

- `tmp/causal_regen_20260516/omega4_6_2_loss_cluster_governor_v5_live_native_val_oos_20260702/report.json`
- `tmp/causal_regen_20260516/omega4_6_borrowed_cap200_live_native_val_oos_20260702/report.json`
- `tmp/causal_regen_20260516/omega4_6_2_cap220_paper_optstop_live_native_val_oos_20260702/report.json`
- `tmp/causal_regen_20260516/omega4_6_2_roll8_two_stage_veto_live_native_val_oos_20260702/report.json`

## Verdict

No candidate is live-promotion eligible from this batch.

The best live-native OOS PnL is `loss_cluster_v5` at +53.36%, but its validation result is -32.05% with -32.26% MDD. That is not robust enough to treat as a validated upgrade.
