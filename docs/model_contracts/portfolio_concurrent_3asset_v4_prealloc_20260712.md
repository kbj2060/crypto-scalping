# Portfolio Concurrent 3-Asset v4 Pre-Allocated Budget - 2026-07-12

Status: `research_diagnostic_not_live_wired`.

Follow-up to `docs/model_contracts/portfolio_concurrent_3asset_v3_soft_cap_20260712.md`, which
found v3's shared-budget soft-scaling cap starved SOL (skipped-below-floor 94/oos_extended vs
ETH 0 / BTC 1) because the fixed `eth, sol, btc` checking order lets ETH always claim first share
of a *shared* notional pool. v4 replaces the shared pool with a fixed, non-competing per-asset
budget, per the user's requested priority order ETH > BTC > SOL:

`--cap-mode prealloc --eth-share 0.5 --btc-share 0.3 --sol-share 0.2` (shares normalized to sum
1; defaults match this request). Each asset's own budget = `total_notional_cap * its share`,
checked independently of what any other asset is doing -- no cross-asset lookup, no checking-order
dependence at all. `same_direction_notional_cap` is unused in this mode: worst-case same-direction
stacking is already structurally bounded by the sum of the three shares (<= total_notional_cap
when all three happen to be open, same direction, at their own max simultaneously).

## Results (all at total_notional_cap=3.0)

| config | split | PnL | realized MDD | MTM MDD | trades |
|---|---|---:|---:|---:|---:|
| v1 uncapped | validation | 164.03% | -29.24% | -35.50% | 84 |
| v1 uncapped | oos_extended | 69.70% | -38.21% | -45.01% | 116 |
| v1 uncapped | oos_frozen_q1_2026 | 83.61% | -38.21% | -45.01% | 70 |
| v3 scale (shared, loose) | validation | 53.30% | -27.79% | -33.92% | 85 |
| v3 scale (shared, loose) | oos_extended | 10.02% | -33.21% | -35.02% | 120 |
| v3 scale (shared, loose) | oos_frozen_q1_2026 | 47.71% | -30.19% | -33.29% | 71 |
| v4 prealloc (ETH50/BTC30/SOL20) | validation | 96.77% | -26.21% | -31.70% | 84 |
| v4 prealloc (ETH50/BTC30/SOL20) | oos_extended | **54.42%** | **-32.45%** | **-38.26%** | 116 |
| v4 prealloc (ETH50/BTC30/SOL20) | oos_frozen_q1_2026 | 62.94% | -32.45% | -38.26% | 70 |

## Starvation check (the reason v4 exists)

| config | split | eth skipped-below-floor | sol skipped-below-floor | btc skipped-below-floor |
|---|---|---:|---:|---:|
| v3 scale (shared) | oos_extended | 0 | **94** | 1 |
| v4 prealloc | oos_extended | 0 | **0** | 0 |

Trade counts under v4 match v1 uncapped exactly on every split (84/116/70) -- every candidate that
would have opened uncapped still opens under prealloc, just at a smaller notional when its own
share-based budget binds (scaling counts: e.g. oos_extended eth=15, sol=27, btc=18 scaled events,
mean scale ratio ~0.72, min ~0.50). No asset is ever crowded out by another.

## Interpretation

v4 dominates v3 on this comparison: better PnL (+54.4% vs +10.0% oos_extended) with comparable or
better MDD reduction (-32.4%/-38.3% vs -33.2%/-35.0%), and zero starvation anywhere. This confirms
the v3 finding was specifically an artifact of the *shared* budget's checking order, not of
capping notional per se -- giving each asset a fixed, guaranteed slice removes the pathology
entirely while keeping the real benefit (materially lower combined drawdown than uncapped).

The ETH 50% / BTC 30% / SOL 20% split was chosen to match the user's explicitly requested priority
order (ETH largest, then BTC, then SOL smallest); it is a deliberate sizing preference, not a
data-derived optimum -- note SOL was actually one of the stronger solo contributors in the
uncapped run (see `docs/model_contracts/portfolio_concurrent_3asset_native_20260712.md` per-asset
aggregates), so this allocation intentionally under-weights SOL relative to its own historical
contribution in exchange for the stated ETH/BTC priority.

## Recommendation

Of the three cap designs tried (reject/scale/prealloc), `prealloc` with fixed per-asset shares is
the only one with no known pathology (no path-dependent substitution like v2, no order-driven
starvation like v3). Treat `--cap-mode prealloc --total-notional-cap 3.0 --eth-share 0.5
--btc-share 0.3 --sol-share 0.2` as the current best-available portfolio-cap candidate for this
architecture, pending: a broader sweep of `total_notional_cap` and share values, and the other
still-open items (duration-gate-off stress test, live-path parity, fresh 2026-07+ window).

## Caveats

- Same modeling caveats as v1/v2/v3 apply: per-asset numbers from the shared ledger are not a
  dedicated-capital replay; new positions size off realized cash only; not a promotion artifact,
  no live wiring.
- Only one share split (50/30/20) was tried. The shares are a stated preference, not something
  this replay searched over.
