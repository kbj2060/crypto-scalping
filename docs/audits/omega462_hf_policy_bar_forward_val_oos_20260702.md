# Omega4.6.2 HF Policy Overlay Bar-Forward Validation/OOS - 2026-07-02

- Model: `omega4_6_2_source_parent_fresh_forward_with_hf_policy_overlay_20260702`
- Runner: `scripts/run_omega462_hf_policy_bar_forward_val_oos_20260702.py`
- Report: `tmp/causal_regen_20260516/omega462_hf_policy_bar_forward_val_oos_20260702/report.json`

## Fresh-Forward Definition

- Mode: fixed-split historical 5m bar-by-bar causal walk-forward.
- Validation: `[2025-09-01 00:00:00, 2026-01-01 00:00:00)`
- OOS: `[2026-01-01 00:00:00, 2026-04-01 00:00:00)`
- Trade ledgers used as input: `false`
- Saved parent entry/exit timestamps used: `false`
- Parent decision cache used: `false`
- Future rows used for entry: `false`
- Fresh-forward bar-by-bar: `true`

## Data Sources

- Validation feature frame: `data/splits/year_oos/training_features_2025.csv`
- OOS feature frame: `tmp/causal_regen_20260516/extended_oos_20260702/training_features_2026_0101_0702_m7_ai_for_omega5_parity.csv`

## Policy Contract

- Source parent: runtime-native Omega4.6.2 source-parent adapter.
- Overlay config: `tmp/causal_regen_20260516/extended_oos_20260702/omega4_6_2_cached_parent_policy_upgrade_hf_papers_fast_20260702/report.json`
- TP price move: `0.026`
- SL price move: `0.012`
- Notional cap: `2.2`
- Max leverage observed: `5.0`
- Same-bar ambiguity: stop-loss first.
- Split-end open positions are not force-closed.

## Results

Validation:
- Compound PnL: `-14.86%`
- Compound MDD: `-25.52%`
- Additive PnL: `-13.12%`
- Trades: `166`
- WR: `29.52%`
- Long/short trades: `32` / `134`
- Avg hold: `11.47h`

OOS:
- Compound PnL: `+80.29%`
- Compound MDD: `-7.64%`
- Additive PnL: `+61.30%`
- Trades: `81`
- WR: `41.98%`
- Long/short trades: `18` / `63`
- Avg hold: `12.14h`

## Integrity Checks

- Validation ledger replay trace count: `0`
- Validation non-native trace count: `0`
- Validation non-`-1` source policy row count: `0`
- OOS ledger replay trace count: `0`
- OOS non-native trace count: `0`
- OOS non-`-1` source policy row count: `0`

## Verdict

- OOS is strong, but validation is negative with MDD below the prior risk tolerance.
- This model is materially stronger than `omega5_live_short_momentum_v2` under the corrected bar-forward definition, but it should not be treated as a clean promotion pass without addressing validation fragility.
