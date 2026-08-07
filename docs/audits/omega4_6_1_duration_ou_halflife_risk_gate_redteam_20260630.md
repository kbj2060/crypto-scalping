# Omega 4.6.1 Duration OU-Halflife Risk Gate Red-Team Record - 2026-06-30

## Verdict

`CONDITIONAL_PASS_MAX_HOLD_AND_PNL_TARGET_EXCLUDED_NOT_DAYTRADING_LIVE_PASS`

This candidate improves Omega4.6 under the same conditional swing baseline
scope. It remains blocked for full day-trading live promotion because max hold
still exceeds 24h.

## Evidence

- Runtime contract: `tmp/causal_regen_20260516/omega4_6_1_duration_ou_halflife_risk_gate_20260630/runtime_contract.json`
- Source report: `tmp/causal_regen_20260516/omega4_6_duration_aware_risk_layer_20260630/report.json`
- Artifact audit: `tmp/causal_regen_20260516/omega4_6_duration_aware_risk_layer_20260630/omega_artifact_integrity_audit_20260630.json`

## Improvements Versus Omega4.6

- Validation PnL: `+117.17%` -> `+175.86%`
- Validation MDD: `-17.43%` -> `-10.60%`
- Validation max hold: `222.00h` -> `115.33h`
- OOS readout PnL: `+67.85%` -> `+72.59%`
- OOS readout MDD: `-13.28%` -> `-7.47%`
- OOS readout max hold: `218.50h` -> `133.50h`

## Remaining Risks

- Trade count falls from validation `29` to `21`, and OOS `13` to `9`.
- The rule skips some profitable trades, including one OOS take-profit.
- The rule is selected from a small validation trade sample. It must be
  treated as a frozen candidate and retested in future walk-forward windows.
- Max hold remains above 24h, so this is not a day-trading live-pass model.

## Required Before Live Wiring

- Confirm `ou_halflife` availability and parity in the live feature frame.
- Add fail-fast runtime feature checks.
- Run runtime-native replay, not only ledger replay.
- Keep OOS excluded from future rule/threshold selection.
