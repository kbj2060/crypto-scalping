# Omega 4.6.1 Duration OU-Halflife Risk Gate Contract - 2026-06-30

## Status

- Model id: `omega4_6_1_duration_ou_halflife_risk_gate_20260630`
- Base model: `omega4_6_plus_t12_nohold_risk1_20260630`
- Status: `conditional_upgrade_candidate_not_live_wired`
- Runtime contract: `tmp/causal_regen_20260516/omega4_6_1_duration_ou_halflife_risk_gate_20260630/runtime_contract.json`
- Candidate manifest: `data/ensemble/supervised/omega4_6_1_duration_ou_halflife_risk_gate_20260630/candidate_manifest.json`
- Promotion manifest: `tmp/causal_regen_20260516/omega4_6_1_duration_ou_halflife_risk_gate_20260630/promotion_manifest.json`
- Diagnostic source: `tmp/causal_regen_20260516/omega4_6_duration_aware_risk_layer_20260630/report.json`

This is a validation-only selected duration-aware entry risk gate. It is not
live-wired and it keeps the Omega4.6 conditional swing classification.

## Rule

At entry time, read `ou_halflife`.

```text
if ou_halflife <= 0.005415348:
    hit_scale = 0.0
else:
    hit_scale = 1.0
```

The rule scales entry notional and leverage only. It does not alter the exit
head, TP/SL, or trade path.

## Selection

- Selection scope: validation-only duration-priority.
- OOS is readout-only and was not used to choose the rule.
- Objective favors validation monthly robustness and max-hold reduction, while
  requiring validation MDD within 20%.

## Metrics

| Split | PnL | MDD | WR | Trades | Skipped | Max Hold | Avg Hold |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Validation | `+175.86%` | `-10.60%` | `61.90%` | `21` | `8` | `115.33h` | `60.37h` |
| OOS readout | `+72.59%` | `-7.47%` | `66.67%` | `9` | `4` | `133.50h` | `59.41h` |

Baseline Omega4.6 reference:

| Split | PnL | MDD | WR | Trades | Max Hold |
| --- | ---: | ---: | ---: | ---: | ---: |
| Validation | `+117.17%` | `-17.43%` | `51.72%` | `29` | `222.00h` |
| OOS readout | `+67.85%` | `-13.28%` | `53.85%` | `13` | `218.50h` |

## Conditional Pass Scope

The candidate passes the non-excluded gates:

- Artifact integrity
- MDD within 20%
- Leverage within 5x
- Notional within 1.8
- Active-trade overlap check
- Accounting consistency
- `notional = margin_fraction * leverage`

Excluded gates remain:

- Max hold 24h
- Validation/OOS PnL target 100%

## Feature Provenance

`ou_halflife` is generated as an entry-time OU mean-reversion half-life feature
in `features/elite.py`. It is not a future label, but it must remain part of
the live feature contract before any runtime wiring.

## Upgrade Notes

This rule improves validation, OOS readout, MDD, and max-hold versus Omega4.6,
but it reduces trade count. Treat it as the current best conditional upgrade
candidate, not as a final live model.
