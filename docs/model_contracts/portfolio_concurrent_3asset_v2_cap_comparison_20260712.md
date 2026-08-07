# Portfolio Concurrent 3-Asset v2 Cap Comparison - 2026-07-12

Status: `research_diagnostic_not_live_wired`.

Compares the uncapped v1 concurrent replay (`docs/model_contracts/portfolio_concurrent_3asset_native_20260712.md`)
against two v2 hard-reject portfolio caps, using
`scripts/replay_portfolio_concurrent_3asset_native_20260712.py --total-notional-cap X --same-direction-notional-cap Y`.

Cap choice was informed by measuring (not guessing) the uncapped run first: when 2+ sleeves are
open, combined notional averages 2.3-2.8x (max ~4.15x) and same-direction (both/all long, or
both/all short) overlap occurs in ~79-80% of overlap bars, so same-direction stacking -- not
diversified offsetting exposure -- explains most of the combined drawdown.

## Results

| config | split | PnL | realized MDD | MTM MDD | trades |
|---|---|---:|---:|---:|---:|
| v1 uncapped | validation | 164.03% | -29.24% | -35.50% | 84 |
| v1 uncapped | oos_extended | 69.70% | -38.21% | -45.01% | 116 |
| v1 uncapped | oos_frozen_q1_2026 | 83.61% | -38.21% | -45.01% | 70 |
| v2 loose (total=3.0, same_dir=2.2) | validation | 46.89% | -21.59% | -23.77% | 75 |
| v2 loose (total=3.0, same_dir=2.2) | oos_extended | **-20.19%** | -44.97% | **-49.45%** | 107 |
| v2 loose (total=3.0, same_dir=2.2) | oos_frozen_q1_2026 | 46.72% | -24.55% | -29.51% | 64 |
| v2 tight (total=2.5, same_dir=1.8) | validation | 35.05% | -14.16% | -19.29% | 74 |
| v2 tight (total=2.5, same_dir=1.8) | oos_extended | 2.72% | -29.35% | -30.01% | 109 |
| v2 tight (total=2.5, same_dir=1.8) | oos_frozen_q1_2026 | -0.44% | -20.85% | -20.38% | 64 |

## Important counterintuitive finding

The "loose" cap did not behave as a naive risk/return trade-off. On `oos_extended`, PnL went
from +69.70% (uncapped) to **-20.19%**, and MTM MDD got **worse** (-45.01% -> -49.45%) despite
strictly less notional being deployed. Both validation and oos_frozen_q1_2026 improved on both
axes under the same cap; only the full oos_extended window inverted.

Root cause: this is a hard-reject cap, not a scheduler. Skipping a candidate because it would
breach the cap does not defer that asset's entry to the next bar with the same signal -- the
asset stays flat and its `_candidate_for_asset` will only fire again on a later bar, generating a
**different, independent trade** (different entry price, different side, different outcome), not
a delayed version of the rejected one. Combined with the fixed asset processing order
(`eth, sol, btc` -- ETH always gets first claim on scarce notional budget each bar), rejecting
entries changes *which* trades each asset ends up taking, not just *how many*. In `oos_extended`
this substitution effect happened to replace some of ETH/SOL/BTC's better trades with worse ones.
The tighter cap (2.5/1.8) recovered most of the risk reduction (MTM MDD -30.01%, close to
validation's -19.29%) while giving up almost all PnL (+2.72%), which is a more "normal" tradeoff
shape but still not attractive on an absolute basis.

**Conclusion: a naive hard-reject notional cap is not a safe drop-in fix.** It can occasionally
make risk *worse*, not just return lower, because of this substitution/path-dependence effect,
and the effect is not currently predictable from the uncapped run's diagnostics alone.

## Recommendation

Do not adopt a hard-reject total/same-direction notional cap as specified here for live use.
Before any v3 attempt, the more promising direction is a **soft cap that scales down** the
candidate's notional to fit the remaining budget rather than fully rejecting it (preserves the
signal's timing/direction, only reduces its size) -- this avoids the path-dependent substitution
effect entirely, since the asset still opens on the same bar with the same side, just smaller.
This has not been implemented or tested. Until then, treat the v1 uncapped concurrent numbers
(`docs/model_contracts/portfolio_concurrent_3asset_native_20260712.md`) as the reference
diagnostic, with the explicit caveat that combined MTM MDD (-45.01% oos_extended) is real and
currently unmitigated -- any live rollout should size the *base* per-asset leverage/notional
constants down rather than relying on this kind of entry-rejection cap.

## Caveats

- `v2 loose`/`v2 tight` runs used the code version before the doc-write-collision fix (both wrote
  to the same fixed `docs/model_contracts/portfolio_concurrent_3asset_native_20260712.md`,
  clobbering each other and the v1 baseline doc in turn); the v1 baseline doc has since been
  regenerated and is correct. Full report.json for each config remains at
  `tmp/causal_regen_20260516/portfolio_concurrent_3asset_native_20260712_v2_{loose,tight}/report.json`.
  Later runs (with `--out-dir` other than the default) now also write their own `report.md`
  alongside `report.json` instead of clobbering the shared doc.
- Not a promotion artifact. No live wiring.
