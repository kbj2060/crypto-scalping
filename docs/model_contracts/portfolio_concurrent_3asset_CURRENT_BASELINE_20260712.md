# Portfolio Concurrent 3-Asset CURRENT BASELINE - 2026-07-12

Status: `research_diagnostic_not_live_wired`. **This doc designates the current reference
configuration for the ETH/SOL/BTC concurrent portfolio replay, superseding the earlier
v1-uncapped / v2-reject / v3-scale / v4-prealloc designations as the default point of comparison
for future work in this line.** It does not change anything about the underlying per-asset
Omega4.6.1 models or `trading_bot.py` -- still a research replay, not a live candidate.

## Configuration

```
scripts/replay_portfolio_concurrent_3asset_native_20260712.py \
  --duration-gate off \
  --eth-notional-multiplier 1.5 \
  --btc-notional-multiplier 1.0 \
  --sol-notional-multiplier 1.0
```

No `total_notional_cap` (uncapped -- `cap_mode` falls back to `scale` with nothing to cap).
Artifacts: `tmp/causal_regen_20260516/portfolio_concurrent_3asset_native_20260712_baseline_eth15x/`.

## Results

| split | PnL | realized MDD | MTM MDD | trades | WR |
|---|---:|---:|---:|---:|---:|
| validation | 22.90% | -36.71% | -44.42% | 88 | 38.64% |
| oos_extended | 292.19% | -31.23% | -39.25% | 118 | 43.22% |
| oos_frozen_q1_2026 | 315.61% | -26.83% | -38.97% | 71 | 45.07% |

## How this was arrived at (chain of findings, in order)

1. **v1 uncapped concurrent replay** (`docs/model_contracts/portfolio_concurrent_3asset_native_20260712.md`):
   first true concurrent (not mutually-exclusive) 3-asset replay. Found ETH/SOL/BTC open together
   65-70% of bars, mostly same-direction.
2. **v2 (hard-reject cap) / v3 (soft-scale cap) / v4 (pre-allocated per-asset budget)**: iterated
   portfolio-level notional caps. v4 (`docs/model_contracts/portfolio_concurrent_3asset_v4_prealloc_20260712.md`)
   was the best of the three -- no known pathology -- with a clean monotonic risk/return frontier
   over `total_notional_cap`.
3. **Duration-gate stress test** (`docs/model_contracts/portfolio_concurrent_3asset_v4_sweep_duration_gate_20260712.md`,
   `..._gate_off_cap_sweep_20260712.md`): found the existing VAL-selected `ou_halflife` duration
   gate is overfit to validation and actively hurts OOS -- disabling it strictly dominates the
   gate-on frontier at every cap level (higher PnL AND lower MDD simultaneously).
4. **ETH notional multiplier sweep** (this doc): under the corrected gate-off, uncapped baseline,
   swept an unconditional ETH notional multiplier (1.0x-3.0x, BTC/SOL held at 1.0x). PnL rises
   monotonically with the multiplier (uncapped 1.0x: oos_extended +188.95% -> 3.0x: +599.73%), but
   **this is a mechanical leverage-scaling effect, not a signal-quality improvement** -- MDD scales
   up in lockstep, and validation performance inverts sign above ~2.0-2.5x (validation PnL: 1.0x
   +28.28% -> 2.0x +12.17% -> 2.5x **-2.39%** -> 3.0x **-18.98%**), proving that chasing the
   highest-PnL multiplier blindly is not robust. **1.5x was chosen as the new baseline** (not the
   PnL-maximizing 3.0x) because it is the point where OOS improves meaningfully over 1.0x
   (oos_extended +189%->+292%) while validation stays solidly positive (+22.9%) and MDD growth is
   still moderate (-29.1%->-31.2% oos_extended realized MDD) rather than the severe degradation
   seen at 2.0x+.

## Important framing for whoever reads this next

- **This is not a "found a better strategy" result in the same sense as the duration-gate finding
  above.** The duration gate was a methodology bug (overfit feature threshold); disabling it is a
  correctness fix. The ETH multiplier is a deliberate **leverage/risk-sizing decision** applied on
  top of a model that already calibrated its own margin_fraction/leverage -- 1.5x means every ETH
  trade now runs at 1.5x whatever notional the risk sidecar itself decided was appropriate. Do not
  conflate the two: one is "fixing a bug," the other is "choosing to take more risk because
  backtested return was higher at that risk level, up to a validated bound."
- Per-asset numbers in the results table are the same non-dedicated-capital shared-ledger
  aggregates used throughout this line of work (see `docs/model_contracts/portfolio_concurrent_3asset_native_20260712.md`
  for the exact caveat) -- not isolated per-asset PnL.
- **Standing caveat, unresolved:** this whole chain (v1->v2->v3->v4->cap sweep->duration-gate
  test->gate-off re-sweep->ETH-multiplier sweep) selected every design choice by repeatedly
  viewing results on the SAME Jan-Jun 2026 OOS window. Each individual replay is fresh-forward
  causal (no ledger leakage), but this many rounds of "look at OOS, adjust, look again" is the
  same "heavily peeked window" hazard flagged elsewhere in this project (e.g. Sigma6's
  2026-03..06 window). **This configuration has not been confirmed on data that was not used to
  select anything above.** Before any live consideration, re-run this exact configuration
  (`--duration-gate off --eth-notional-multiplier 1.5`) on a genuinely fresh 2026-07+ window and
  confirm the numbers hold up.

## Caveats

- Only ETH's multiplier was swept; BTC/SOL multipliers were held at 1.0 throughout, untested.
- No `total_notional_cap` is applied in this baseline (uncapped) -- combining the ETH multiplier
  with the v4 `prealloc` cap design has not been tested.
- Not a promotion artifact. No live wiring.
