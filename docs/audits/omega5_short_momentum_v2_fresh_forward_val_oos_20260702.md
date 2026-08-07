# Omega5 Short Momentum V2 Fresh-Forward Validation/OOS - 2026-07-02

## Status

- INVALIDATED: this run used a live-only interpretation of fresh-forward.
- Corrected project definition: fresh-forward means fixed-split historical causal walk-forward, processing 5m bars one by one.
- Required split for the corrected test: validation `2025-09-01` to `2025-12-31`, OOS `2026-01-01` to `2026-03-31`.
- The process `2138247` was stopped; trading bot data collection was not stopped.

- Candidate: `omega5_live_short_momentum_v2`
- Runner: `scripts/run_omega5_short_momentum_fresh_forward_val_oos_20260702.py`
- Output directory: `data/live/omega5_live_short_momentum_v2_fresh_forward_val_oos_20260702/`
- Process PID: `2138247`

## Protocol

- Validation window: `2026-07-02T19:02:49+09:00` to `2026-07-03T01:02:49+09:00`
- OOS window: `2026-07-03T01:02:49+09:00` to `2026-07-03T07:02:49+09:00`
- Source: `data/live/decision_feature_snapshot.jsonl`
- Source offset at start: `119979258`
- Historical replay used for selection: `false`
- Trade ledgers used as input: `false`
- Saved parent exit timestamps used: `false`
- Live forward only: `true`

## Candidate Contract

- Side: short-only
- TP price move: `0.0045`
- SL price move: `0.0030`
- Max hold: `25m`
- Notional: `1.0`
- Round-trip cost per notional: `0.0006`

## Initial Readout

- Phase: `collecting_validation`
- Validation decision rows: `1`
- Validation entry rows: `0`
- Validation closed trades: `0`
- OOS closed trades: `0`
- Promotion evidence allowed: `false` until the fresh-forward run completes.
