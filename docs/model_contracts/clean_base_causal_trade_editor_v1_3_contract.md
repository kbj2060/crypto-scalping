# Clean Base Causal Trade Editor V1.3 Contract

Status: `experimental_challenger`

## Purpose

Preserve clean base/Lifecycle core entries and sides while learning a causal per-trade scale and early-exit schedule.

## Runtime Inputs

`side, quality, confidence, core_notional, leverage, funding_abs, funding_pressure, liquidity_vacuum, amihud_illiquidity_z, m7_tail_risk, evt_tail_flag, ai_adverse_risk`

Runtime decisions also use closed-equity account drawdown and daily drawdown gates. They do not use future realized returns, future high/low, or event candidate labels.

## Outputs

- `effective_notional`
- `effective_exit_idx`
- action reason code
- trade-level ledger with cash before/after and prediction telemetry

## Promotion Reference

- Clean base: PnL `177.329809%`, MDD `-17.759665%`
- Causal sleeve v1.2: PnL `210.491277%`, MDD `-18.015155%`
