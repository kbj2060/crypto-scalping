# Portfolio Concurrent 3-Asset v3 Soft-Scaling Cap - 2026-07-12

Status: `research_diagnostic_not_live_wired`.

Follow-up to `docs/model_contracts/portfolio_concurrent_3asset_v2_cap_comparison_20260712.md`,
which found that a hard-reject notional cap is not a safe drop-in fix (it can make both PnL and
MDD worse in some windows, via a substitution/path-dependence effect). This implements the
recommended v3 direction: `--cap-mode scale` in
`scripts/replay_portfolio_concurrent_3asset_native_20260712.py` shrinks a candidate's notional to
fit the remaining budget (holding margin_fraction fixed, reducing leverage) instead of rejecting
it outright, preserving entry timing and side. Only skipped if the capped notional would fall
below `--min-notional` (default 0.05, a dust floor).

## Results: v1 uncapped vs v2 reject vs v3 scale (same thresholds: total=3.0, same_dir=2.2)

| config | split | PnL | realized MDD | MTM MDD | trades |
|---|---|---:|---:|---:|---:|
| v1 uncapped | validation | 164.03% | -29.24% | -35.50% | 84 |
| v1 uncapped | oos_extended | 69.70% | -38.21% | -45.01% | 116 |
| v1 uncapped | oos_frozen_q1_2026 | 83.61% | -38.21% | -45.01% | 70 |
| v2 reject (loose) | validation | 46.89% | -21.59% | -23.77% | 75 |
| v2 reject (loose) | oos_extended | **-20.19%** | -44.97% | **-49.45%** | 107 |
| v2 reject (loose) | oos_frozen_q1_2026 | 46.72% | -24.55% | -29.51% | 64 |
| v3 scale (loose) | validation | 53.30% | -27.79% | -33.92% | 85 |
| v3 scale (loose) | oos_extended | 10.02% | -33.21% | -35.02% | 120 |
| v3 scale (loose) | oos_frozen_q1_2026 | 47.71% | -30.19% | -33.29% | 71 |

**v3 behaves like a real risk dial.** Unlike v2, MDD improves or holds steady on every single
split (realized MDD: -29→-28%, -38→-33%, -38→-30%; MTM MDD: -36→-34%, -45→-35%, -45→-33%) and PnL,
while lower than uncapped, stays positive everywhere (validation +53%, oos_extended +10%,
oos_frozen_q1 +48%) -- no sign flips, no MDD-gets-worse pathology. This confirms the v2 diagnosis:
the problem was hard rejection substituting a different, worse trade, not the cap threshold
itself.

## New finding: fixed processing order starves SOL under this cap

Scaling/skip diagnostics for `oos_extended`:

| asset | scaled events | skipped below floor (0.05) |
|---|---:|---:|
| eth | 16 | 0 |
| sol | 10 | **94** |
| btc | 6 | 1 |

SOL is skipped-below-floor far more than ETH or BTC. Because the open pass always checks
`eth, sol, btc` in that fixed order, ETH claims first share of the remaining total-notional/
same-direction budget every bar; by the time SOL is checked, remaining budget is often already
below the 0.05 floor, so SOL frequently can't open at all under this cap, even though SOL was one
of the stronger solo contributors in the uncapped run (see
`docs/model_contracts/portfolio_concurrent_3asset_native_20260712.md` per-asset aggregates).
(Note: because SOL stays flat while starved, `_candidate_for_asset` re-fires on subsequent bars
for what can be the same underlying multi-bar signal window, so the raw skip count is not a count
of independent lost opportunities -- but the qualitative starvation is real and asset-order-driven.)

**This is a genuine design gap, not yet fixed here**: the cap budget is allocated first-come,
first-served by a hardcoded asset order, not by each asset's actual contribution/quality. A
fairer v4 would pre-allocate cap budget per asset (e.g., a percentage split reflecting each
sleeve's own solo Sharpe/return contribution) or rotate the checking order, rather than giving ETH
permanent first claim.

## Recommendation

Prefer `--cap-mode scale` (the new default) over `--cap-mode reject` if any hard cap is used live
-- it is a monotonic, predictable risk/return dial rather than a path-dependent gamble. The
specific thresholds tried here (total=3.0, same_dir=2.2) are a reasonable middle ground (keeps
MDD in the -30 to -35% range, keeps all splits solidly positive) but the SOL-starvation issue
above should be addressed (fair budget allocation, not fixed order) before treating any specific
threshold as a live recommendation.

## Caveats

- Same modeling caveats as the v1/v2 docs apply: per-asset numbers from the shared ledger are not
  a dedicated-capital replay; new positions size off realized cash only; not a promotion artifact,
  no live wiring.
- Only one threshold pair was tried for v3 (matching v2's "loose" config for direct comparison).
  A broader sweep, and the fair-allocation fix above, are natural next steps.
