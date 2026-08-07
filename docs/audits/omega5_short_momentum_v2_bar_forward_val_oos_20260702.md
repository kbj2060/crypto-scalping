# Omega5 Short Momentum V2 Bar-Forward Validation/OOS - 2026-07-02

- Candidate: `omega5_live_short_momentum_v2`
- Runner: `scripts/run_omega5_short_momentum_bar_forward_val_oos_20260702.py`
- Report: `tmp/causal_regen_20260516/omega5_short_momentum_v2_bar_forward_val_oos_20260702/report.json`

## Fresh-Forward Definition

- Mode: fixed-split historical 5m bar-by-bar causal walk-forward.
- Validation: `[2025-09-01 00:00:00, 2026-01-01 00:00:00)`
- OOS: `[2026-01-01 00:00:00, 2026-04-01 00:00:00)`
- Trade ledgers used as input: `false`
- Saved parent exit timestamps used: `false`
- Future rows used for entry: `false`
- Fresh-forward bar-by-bar: `true`

## Data Sources

- Validation feature frame: `data/splits/year_oos/training_features_2025.csv`
- OOS feature frame: `tmp/causal_regen_20260516/extended_oos_20260702/training_features_2026_0101_0702_m7_ai_for_omega5_parity.csv`

## Candidate Contract

- Side: short-only
- Execution: single-position, no pyramiding
- TP price move: `0.0045`
- SL price move: `0.0030`
- Max hold: `25m`
- Notional: `1.0`
- Round-trip cost per notional: `0.0006`

## Results

Validation:
- Additive PnL: `-327.76%`
- Compound PnL: `-96.32%`
- Compound MDD: `-96.35%`
- Trades: `5910`
- WR: `39.36%`
- Avg hold: `18.61m`

OOS:
- Additive PnL: `-96.96%`
- Compound PnL: `-62.43%`
- Compound MDD: `-62.43%`
- Trades: `2243`
- WR: `41.77%`
- Avg hold: `18.64m`

## Verdict

- `omega5_live_short_momentum_v2` does not pass the corrected fresh-forward validation/OOS test.
- The model fires too frequently on this split and the short-only rule is not profitable after cost.
