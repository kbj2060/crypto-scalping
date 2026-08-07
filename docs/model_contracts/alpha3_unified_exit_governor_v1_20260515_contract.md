# Alpha3 Unified Exit Governor v1 Contract

## Scope

- Entry stack is frozen Alpha3 corrected live contract.
- Entry contract: `alpha3_corrected_selected_touch0_skip_entry`, next-open touch0 maker attempt, skipped if entry maker does not fill.
- This layer is reduce-only. It cannot open, flip, add, or increase a position.
- It governs both parent-owned `v21_2` and deep-scout-owned `deep_alpha` positions with one state/action interface.

## Decision Order

1. Active position state is built each bar after entry.
2. Unified exit governor may choose `hold` or a full reduce-only close with a selected maker-limit exit arm.
3. If the learned layer does not close, existing TP/SL/max-hold checks remain as fallback safety rails.
4. If a safety rail fires, the governor still selects the exit placement arm, but cannot veto the safety exit.

## Selected Runtime

```json
{
  "name": "fixed_touch0_exit_fallback",
  "q_margin": 99.0,
  "min_advantage_conf": 99.0,
  "min_hold": 999,
  "exit_fallback_arm": "baseline_exit2_pen05",
  "force_exit_mode": "fallback"
}
```

## Audit Boundaries

- Train: 2025-01-01 through 2025-09-30.
- Runtime selection: 2025-10-01 through 2025-12-31.
- Fixed OOS: 2026 only after selection.
- Backtest execution still uses 5m OHLC touch proxy, not real queue position.
