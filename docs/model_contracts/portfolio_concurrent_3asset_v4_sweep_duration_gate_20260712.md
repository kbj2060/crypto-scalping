# Portfolio Concurrent 3-Asset v4 Cap Sweep + Duration-Gate Stress Test - 2026-07-12

Status: `research_diagnostic_not_live_wired`.

Two follow-ups to `docs/model_contracts/portfolio_concurrent_3asset_v4_prealloc_20260712.md`:
(1) a broader sweep of `total_notional_cap` and per-asset share values in `prealloc` mode, and
(2) a stress test of the `ou_halflife` duration gate (monkeypatch `native.DURATION_THRESHOLDS` to
disabled, i.e. the gate never blocks an entry). Script:
`scripts/sweep_portfolio_concurrent_3asset_v4_20260712.py`. World is built once per split and
replayed many times in-process against the same causal data (no re-fetching/re-scoring).

## 1. total_notional_cap sweep (shares fixed at eth50/btc30/sol20)

| total_notional_cap | VAL PnL | VAL MDD | oos_extended PnL | oos_extended MDD | oos_extended MTM MDD | Q1 PnL | Q1 MDD | trades |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| uncapped | 164.03% | -29.24% | 69.70% | -38.21% | -45.01% | 83.61% | -38.21% | 116 |
| 1.5 | 38.23% | -20.86% | 25.55% | -18.13% | -21.53% | 27.48% | -18.13% | 116 |
| 2.0 | 55.87% | -22.96% | 34.86% | -23.54% | -27.79% | 38.05% | -23.54% | 116 |
| 2.5 | 74.64% | -24.94% | 43.05% | -28.40% | -33.41% | 49.48% | -28.40% | 116 |
| 3.0 | 96.77% | -26.21% | 54.42% | -32.45% | -38.26% | 62.94% | -32.45% | 116 |
| 3.5 | 114.16% | -27.52% | 62.51% | -36.30% | -42.81% | 71.95% | -36.30% | 116 |
| 4.0 | 128.60% | -28.30% | 66.52% | -37.81% | -44.74% | 77.06% | -37.81% | 116 |

**This is a clean, smooth, monotonic risk/return frontier** on every split -- PnL and MDD both
increase together as the cap loosens from 1.5 to 4.0, with zero pathology (unlike v2's
hard-reject, which could invert PnL sign and worsen MDD simultaneously). Trade count is identical
(116) at every cap level -- the `min_notional=0.05` dust floor is never binding even at the
tightest cap tried, because each asset's own pre-allocated share stays well above it
(e.g. SOL's floor at total=1.5 is `1.5*0.2=0.30`, still 6x the dust floor). This confirms `prealloc`
behaves as a genuine, predictable risk dial across its full useful range. Pick a point on this
table based on risk tolerance; there is no data-derived "best" point, only a frontier.

## 2. asset_shares sweep (total_notional_cap fixed at 3.0)

| shares (eth/btc/sol) | VAL PnL | VAL MDD | oos_extended PnL | oos_extended MDD | oos_extended MTM MDD | Q1 PnL | Q1 MDD |
|---|---:|---:|---:|---:|---:|---:|---:|
| 50/30/20 (user's requested order) | 96.77% | -26.21% | 54.42% | -32.45% | -38.26% | 62.94% | -32.45% |
| 40/35/25 | 102.50% | -26.51% | 49.70% | -31.15% | -37.40% | 56.80% | -31.15% |
| equal 33/33/33 | 118.34% | -26.11% | 48.30% | -29.29% | -34.85% | 57.37% | -29.29% |
| 60/25/15 (more ETH-heavy) | 83.16% | -25.98% | 54.41% | -33.61% | -39.05% | 62.71% | -33.61% |

Share choice has a real but secondary effect (smaller than the total-cap level itself). Equal
weighting gives the best oos_extended MDD (-29.29%/-34.85%) of the four tried, at a modest PnL
cost vs 50/30/20 (48.30% vs 54.42%). Pushing ETH's share up further (60/25/15) does not improve
oos_extended PnL over 50/30/20 (54.41% vs 54.42%, essentially flat) but does worsen MDD slightly
(-33.61% vs -32.45%) -- there are diminishing/negative returns to over-concentrating in ETH beyond
~50%. The user's requested 50/30/20 remains a reasonable choice, not the Sharpe-optimal one, which
looks closer to equal-weighting on this data; this is presented as information, the priority order
was an explicit user preference and is left as-is unless the user wants to revisit it.

## 3. Duration-gate (`ou_halflife`) stress test -- IMPORTANT FINDING

Reference config: `total_notional_cap=3.0`, shares eth50/btc30/sol20. Compares the current
VAL-selected `native.DURATION_THRESHOLDS` (gate on) against those thresholds disabled entirely
(gate off, i.e. the entry-time `ou_halflife` filter never blocks a candidate).

| split | gate ON: PnL | gate ON: MDD | gate ON: trades | gate OFF: PnL | gate OFF: MDD | gate OFF: trades |
|---|---:|---:|---:|---:|---:|---:|
| validation | 96.77% | -26.21% | 84 | 12.37% | -26.45% | 88 |
| oos_extended | 54.42% | -32.45% | 116 | **139.72%** | **-21.00%** | 118 |
| oos_frozen_q1_2026 | 62.94% | -32.45% | 70 | **143.19%** | **-15.99%** | 71 |

**The gate helps validation and hurts OOS badly -- the opposite of what a duration gate should do
if it generalized.** Removing it roughly *doubles* PnL on both OOS splits (54%→140%, 63%→143%)
while cutting MDD by a third to a half (-32%→-21%, -32%→-16%), with only 1-2 extra trades (not a
fundamentally different trade population). On validation the effect inverts: PnL collapses from
+96.77% to +12.37% with almost unchanged MDD (-26.21% vs -26.45%) and only 4 extra trades --
meaning the handful of trades the gate blocks on validation are disproportionately bad ones, while
the trades it blocks on OOS are disproportionately *good* ones.

This corroborates a pre-existing project concern (memory: `project-omega4-6-1-extended-oos-retest`
-- "`ou_halflife` feature has near-zero correlation vs original scoring, `features/elite.py` drift")
with new, concrete, fresh-forward evidence in this concurrent-portfolio context specifically: the
current thresholds (`DURATION_THRESHOLDS = {"eth": 0.005417, "sol": 0.0055208323, "btc":
0.00541154875}`, VAL-selected per-asset in earlier solo work) look like classic validation
overfitting when evaluated causally out-of-sample here.

**This is a bigger lever than any notional cap explored in this whole v1-v4 line of work.**
Doubling OOS PnL and nearly halving OOS MDD by removing one feature gate dwarfs the effect of
tuning `total_notional_cap` or asset shares. Before further cap-tuning, the duration gate itself
should be re-examined: either drop it, or re-derive/re-validate it on a scheme that doesn't overfit
val-vs-oos this severely (e.g. re-select on a proper walk-forward split, or replace the point
threshold with something less brittle).

## Caveats

- Only a full on/off comparison was run, not a re-optimization of the threshold itself -- it's
  possible some other threshold (not zero, not the current VAL-selected one) performs well on both
  VAL and OOS; that hasn't been tested.
- Same modeling caveats as v1-v4 apply throughout (shared-ledger per-asset numbers are not
  dedicated-capital; not a promotion artifact; no live wiring).
- The cap sweeps above (sections 1-2) were run WITH the (apparently overfit) duration gate ON,
  since that was the existing default; if the gate is dropped, the whole v1-v4 cap-tuning
  exercise should probably be re-run on the gate-off baseline, since the underlying PnL/MDD
  numbers it was tuning against would change substantially.
