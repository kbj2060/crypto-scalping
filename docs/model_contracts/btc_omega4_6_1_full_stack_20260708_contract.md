# BTC Omega4.6.1 Full-Stack Replication — single-component h48qual candidate

Status: `research_positive_signal_not_live_wired`.

This is the BTC analogue of the SOL Omega4.6.1 replication flow. It uses BTC-specific raw data,
FeatureEngineer output, regime3-current HMM overlay, BTC-specific parent predictions, ATR
TP/SL, learned exit-head replay, risk sidecar sizing, final scale-map, and a VAL-only
`ou_halflife` duration gate. It is not wired into `trading_bot.py`.

## Selected candidate

- Component: `h48qual`
- Parent quality threshold: `q055` / `0.55`
- Risk sidecar: `tmp/causal_regen_20260516/btc_omega4_2_trade_risk_sidecar_20260708_h48qual_q055_20260708/risk_sidecar.pkl`
- Parent prediction dir: `tmp/causal_regen_20260516/btc_omega4_3head_parent72_loose_entry_quality_20260708_h48qual_20260708`
- Final scale-map: `long_scale=0.5`, `short_scale=2.5`
- Caps: `leverage_cap=5.0`, `notional_cap=1.8`
- Exit threshold: `0.95`
- Duration gate selected on VAL: `ou_halflife > 0.00541154875`

The naive VAL-best fast search selected `zig075 q065`, but it failed OOS badly. The practical
candidate applies a minimum VAL trade-count screen and selects `h48qual q055` instead.

## Exact final replay

Script:

`scripts/apply_final_scale_map_btc_20260708.py`

Report:

`tmp/causal_regen_20260516/btc_final_scale_map_20260708/report.json`

No duration gate:

| split | PnL | MDD | trades | WR |
|---|---:|---:|---:|---:|
| VAL | +7.45% | -11.93% | 16 | 31.25% |
| OOS extended | +22.69% | -15.88% | 30 | 40.00% |

With selected duration gate:

| split | PnL | MDD | trades | WR |
|---|---:|---:|---:|---:|
| VAL | +12.39% | -6.49% | 10 | 40.00% |
| OOS extended | +29.23% | -10.65% | 24 | 41.67% |
| OOS frozen Q1 2026 | +10.17% | -10.65% | 16 | 37.50% |

Replay flags:

- `fresh_forward_bar_by_bar=true`
- `trade_ledgers_used_as_input=false`
- `saved_parent_exit_timestamps_used=false`
- `future_rows_used_for_entry=false`

## Caveats

- Trade count is thin, especially after duration gating.
- This is a single-component BTC candidate, not ETH's full `h48qual + zig075` greedy router.
- No live adapter or `trading_bot.py` wiring was added.
