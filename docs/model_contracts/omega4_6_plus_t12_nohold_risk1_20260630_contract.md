# Omega 4.6 plus_t12 No-Hold Risk1 Contract - 2026-06-30

## Status

- Model id: `omega4_6_plus_t12_nohold_risk1_20260630`
- Display version: `Omega 4.6 conditional swing/runner baseline`
- Status: `current_omega_research_baseline_not_live_wired_conditional_swing`
- Role: Omega4.6 research/upgrade baseline for swing/runner successor work
- Red-team verdict: `CONDITIONAL_PASS_MAX_HOLD_AND_PNL_TARGET_EXCLUDED_NOT_DAYTRADING_LIVE_PASS`
- Live wiring: unchanged

This is not a day-trading or full live PASS model. It is a conditional swing
baseline: max-hold and PnL target are excluded from the mandatory red-team
gates. Successor work must preserve artifact integrity and reduce tail hold
time without applying a blunt post-hoc 24h time stop.

## Lineage

1. Source family: `plus_t12`.
2. Source report:
   `tmp/causal_regen_20260516/omega_creative_until_10am_20260630/plus_t12_diagnostic_nohold_risk1_20260630/report.json`.
3. Router order: `h48qual > zig075`.
4. Selection basis: original `plus_t12_target_guard_03` scale/router contract,
   rebuilt from exact-threshold precomputed component artifacts.
5. OOS remains readout-only for successor selection.

## Runtime Contract

```text
notional = margin_fraction * leverage
PnL = realized_price_move * notional
```

- Leverage cap: `5.0`
- Notional cap: `1.8`
- Live risk scale: `1.0`
- Max hold hours in this no-hold baseline: `0.0`
- Notional-scaled SLTP: `false`

## Components

| Component | Quality threshold | Prediction tag | Precomputed prediction dir |
| --- | ---: | --- | --- |
| `h48qual` | `0.50` | `q050` | `tmp/causal_regen_20260516/omega4_3head_parent72_loose_entry_quality_20260620_zigzagfix_06_h48_quality_noctx_padded_e2_fulltrain_exit30k_20260630` |
| `zig075` | `0.75` | `q075` | `tmp/causal_regen_20260516/omega4_3head_parent72_loose_entry_quality_20260620_current_only_alllabels_01_zigzag_action_labels_20260531_e2_fulltrain_exit30k_20260629` |

Each component must keep exact-threshold parent prediction artifacts:
`train_predictions_qXXX.csv`, `validation_predictions_qXXX.csv`, and
`oos_predictions_qXXX.csv`. The risk sidecar report must keep
`risk_model.precomputed_prediction_dir` and
`risk_model.precomputed_prediction_tag`.

## Scale Map

```json
{
  "h48qual_L": 0.38,
  "h48qual_S": 2.499,
  "zig075_L": 2.446,
  "zig075_S": 2.478
}
```

## Metrics

| Split | PnL | MDD | WR | Trades | Avg notional | Max leverage | Max notional | Max hold |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Validation | `+117.17%` | `-17.43%` | `51.72%` | `29` | `1.4116` | `5.0` | `1.8` | `222.0h` |
| OOS readout | `+67.85%` | `-13.28%` | `53.85%` | `13` | `1.4080` | `5.0` | `1.8` | `218.5h` |

## Passed Non-Excluded Gates

- Artifact integrity audit: `promotion_pass=true`
- Validation and OOS MDD within `20%`
- Leverage within `5x`
- Notional within `1.8`
- No selected-trade overlap
- Accounting consistency
- Notional contract consistency: `notional = margin_fraction * leverage`

## Known Blockers For Full Live/Day Trading

- Validation max hold: `222.0h`, with `21/29` trades over 24h.
- OOS max hold: `218.5h`, with `12/13` trades over 24h.
- OOS PnL is below a `+100%` target.
- A 24h forced time-stop changes the exit contract and destroys validation PnL.

## Upgrade Priorities

1. Preserve the no-hold swing alpha and reduce tail hold time without blunt 24h
   time-stop.
2. Search exit, partial-profit, and trailing-giveback policies that shorten
   holding while preserving TP runners.
3. Improve OOS toward `100%` without using OOS for selection; use
   validation-only selection and then blind OOS readout.
4. Keep futures sizing explicit:
   `notional = margin_fraction * leverage`; `PnL = price_move * notional`.

## Artifacts

- Runtime contract:
  `tmp/causal_regen_20260516/omega4_6_plus_t12_nohold_risk1_20260630/runtime_contract.json`
- Promotion manifest:
  `tmp/causal_regen_20260516/omega4_6_plus_t12_nohold_risk1_20260630/promotion_manifest.json`
- Candidate manifest:
  `data/ensemble/supervised/omega4_6_plus_t12_nohold_risk1_20260630/candidate_manifest.json`
- Source report:
  `tmp/causal_regen_20260516/omega_creative_until_10am_20260630/plus_t12_diagnostic_nohold_risk1_20260630/report.json`
- Artifact audit:
  `tmp/causal_regen_20260516/omega_creative_until_10am_20260630/plus_t12_diagnostic_nohold_risk1_20260630/omega_artifact_integrity_audit_20260630.json`
- Red-team record:
  `docs/audits/omega4_6_plus_t12_nohold_risk1_redteam_20260630.md`
- Handoff:
  `docs/subagents/omega4_6_handoff_20260630.md`
