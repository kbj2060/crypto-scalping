# Clean Base Conformal Downside Filter V1.4 Contract

Status: `experimental_challenger`

## Purpose

Preserve clean base/Lifecycle entries and sides while applying validation-calibrated downside-only shrink and early-exit controls.

## Runtime Inputs

`side, quality, confidence, core_notional, leverage, funding_abs, funding_pressure, liquidity_vacuum, amihud_illiquidity_z, m7_tail_risk, evt_tail_flag, ai_adverse_risk`

Runtime decisions also use closed-equity account drawdown, daily drawdown, and a validation residual quantile chosen before OOS evaluation.

## Forbidden Runtime Inputs

`evt_candidate_side, evt_candidate_label, evt_side_margin, future high/low/close`

## Outputs

- `effective_notional`, constrained to be no larger than the clean base core notional
- `effective_exit_idx`, constrained to be no later than the clean base/Lifecycle core exit
- action reason code
- trade-level ledger with cash before/after, fee cash, predicted downside telemetry, and conformal lower confidence bound

## Split Contract

- Train labels: 2025 pre-validation window
- Calibration/selection: 2025 validation window
- OOS: 2026, one-shot after threshold selection
