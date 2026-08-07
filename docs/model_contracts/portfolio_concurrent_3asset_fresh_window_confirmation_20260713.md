# Portfolio Concurrent 3-Asset — Fresh Window Confirmation — 2026-07-13

Status: `research_diagnostic_not_live_wired`. Answers the standing caveat left open by
[`portfolio_concurrent_3asset_CURRENT_BASELINE_20260712.md`](portfolio_concurrent_3asset_CURRENT_BASELINE_20260712.md):
that doc's entire design chain (v1→v2→v3→v4→duration-gate test→ETH-multiplier sweep) repeatedly
looked at the same 2026-01-01..06-30 OOS window, so it explicitly asked for confirmation on a
genuinely fresh window before any live consideration.

## What was done

1. Extended raw data (5m klines, funding rate, OI/top-trader metrics) for ETH/SOL/BTC from their
   prior cutoffs (~2026-07-01/07) through 2026-07-12, via the public Binance REST/data.binance.vision
   endpoints (no live-account client touched).
2. Rebuilt ETH's `training_features_2026_rebuilt.csv` (via the existing `scripts/update_features.py`
   pipeline) and SOL/BTC's `{sol,btc}_features_2026.csv` (via `FeatureEngineer`, full recompute not
   append-only, to avoid `ou_halflife`/`garch_vol_z` rolling-window seeding errors) through
   2026-07-12.
3. Re-transformed the regime3-current-wide24 HMM overlay for all three assets on the extended range
   (frozen 2024-trained joblib, causal `_transform`, no retraining). **Known, pre-diagnosed
   deviation**: the reproducibility gate in `apply_regime3_wide24_sidecar_extended_20260704.py`
   (byte-match vs the existing Jan-Jun sidecar) fails on this extension (max diff ~0.68) because
   `ou_halflife`/`garch_vol_z` legitimately changed formula upstream sometime after that sidecar was
   built (already found and attributed to a genuine fix, not a bug, during the 2026-07-04 session --
   see `omega6_synthesis_v1_20260703_contract.md`). A new script,
   `scripts/apply_regime3_wide24_sidecar_extended_20260713.py`, warns instead of aborting on this
   specific known mismatch and proceeds with the fresh (current-formula) transform as authoritative.
4. Rebuilt SOL/BTC's wave3 zigzag direction labels and BTC's h48-padded quality labels for the
   extended 2026 range (same recipe, unchanged hyperparameters).
5. Re-scored (not retrained) all three frozen parent bundles' `oos_predictions_qXXX.csv` on the
   extended frame: ETH via the existing `build_omega4_6_1_extended_parent_predictions_20260706.py`
   (already pointed at the files this session extended), SOL/BTC via a new
   `scripts/rescore_sol_btc_parent_predictions_20260713.py`.
6. Re-ran the exact frozen CURRENT_BASELINE config (`--duration-gate off --eth-notional-multiplier
   1.5`, uncapped, `cap_mode=scale`) via a new wrapper,
   `scripts/replay_portfolio_fresh_window_20260713.py`, which adds a lower-bound `entry_floor`
   variant of `_replay_concurrent` (the existing script only has an upper-bound `entry_cutoff`, used
   for the Q1 sub-split) so only entries at/after 2026-07-01 are allowed, while still simulating the
   full window bar-by-bar (no re-simulation-from-scratch artifact).

## Result

| window | PnL | realized MDD | MTM MDD | trades | WR |
|---|---:|---:|---:|---:|---:|
| full extended OOS (2026-01-01..07-12, for comparison) | 62.63% | -46.08% | -46.94% | 127 | 40.94% |
| **fresh window (2026-07-01..07-12 only)** | **+43.63%** | **-1.12%** | **-16.53%** | **7** | **100%** |

Per-asset fresh-window trades: ETH 2 (WR 100%), SOL 3 (WR 100%), BTC 2 (WR 100%).

## Honest read

**Directionally positive, but this is not a meaningful statistical confirmation** -- 7 trades over
12 calendar days is far too small a sample to validate anything, consistent with this whole model
family's standing trade-count-scarcity caveat (see `[[project-sigma8-sigma9-failed-attempts]]`,
ETH's own 24-trade OOS, etc.). The 100% win rate is more likely a small-sample artifact than a real
property. What this genuinely establishes: the frozen baseline did NOT immediately blow up or
invert sign on the first truly-unseen data it encountered, which is a real (if weak) signal --
compare to the actual portfolio-combination-layer experiments in this project, several of which DID
immediately invert sign or collapse to a no-op the moment they hit real fresh/held-out data. This
result does not replace the need to keep accumulating fresh-forward evidence over a longer unpeeked
window before any live consideration.

## No live wiring

No `trading_bot.py` changes. Purely a research confirmation step.
