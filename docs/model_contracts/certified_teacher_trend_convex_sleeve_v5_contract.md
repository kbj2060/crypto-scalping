# Certified Teacher Trend Convex Sleeve V5

- Model ID: `certified_teacher_trend_convex_sleeve_v5`
- Candidate universe: rare high-conviction trend events, not every bar.
- Inputs: certified AI meta, M7 meta, clean_regime_2024_unsup_v4, and causal market/microstructure features.
- Execution: next-bar open entry, convex trend contracts with loose TP and trailing exits.
- Selection: 2025 selection only. 2026 is fixed OOS and not used for config selection.
- Audit: `pass`
- Blocking: `[]`

## Selected Config
- Config: `{'top_k_per_day': 1, 'min_event_score': 0.7, 'min_pred_edge_pct': 0.0, 'max_notional': 1.2, 'min_notional': 0.2, 'min_gap_bars': 36, 'max_pred_adverse_pct': 1.8, 'leverage': 5.0}`

## OOS Cost1
- PnL: `-5.255196299907416`
- MDD: `-5.995319427911394`
- Trades/day: `1.0056818181818181`
- Avg notional: `0.35076591118357525`
