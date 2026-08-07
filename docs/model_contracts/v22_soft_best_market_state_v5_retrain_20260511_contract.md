# V22 Soft Best Market State V5 Retrain

- Model ID: `v22_soft_best_market_state_v5_retrain_20260511`
- Purpose: replacement retrain for deleted V22 soft-best clean artifact, using 2024-only unsupervised market-state features and no legacy regime inputs.
- Audit: `pass`
- Blocking: `[]`
- Train fit: `2025-01-01 00:00:00` to `2025-08-31 23:55:00`
- Selection: `2025-09-01 00:00:00` to `2025-10-31 23:55:00`
- Holdout: `2025-11-01 00:00:00` to `2025-12-31 20:50:00`
- OOS: `2026-01-01 00:00:00` to `2026-02-28 16:00:00`

## Cost1 OOS
- PnL: `-15.201080436847104`
- MDD: `-18.976601891082932`
- Trades: `207`
- Trades/day: `3.5284090909090913`

## Runtime
- Selected config: `{'threshold': 0.58, 'gap': 0.06, 'max_notional': 1.0, 'min_notional': 0.35, 'leverage': 5.0, 'max_hold_bars': 36, 'stop_loss': 0.012, 'take_profit': 0.035, 'trailing_stop': 0.011, 'cooldown_bars': 2, 'state_conf_floor': 0.24, 'risk_off_cap': 0.92, 'candidate_stride': 8}`
- Feature count: `118`
- Market state feature count: `17`

## Warning Resolution
- The OOS ledger includes a `coverage_end` sentinel at the final 2026 eval timestamp, so the old `eval_window_extends_beyond_available_v22_sniper_ledger` warning is not applicable to this regenerated artifact.
