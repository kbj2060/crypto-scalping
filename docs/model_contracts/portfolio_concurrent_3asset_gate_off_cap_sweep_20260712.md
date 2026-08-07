# Portfolio Concurrent 3-Asset Cap Sweep on Duration-Gate-OFF Baseline - 2026-07-12

Status: `research_diagnostic_not_live_wired`.

Direct follow-up to `docs/model_contracts/portfolio_concurrent_3asset_v4_sweep_duration_gate_20260712.md`,
which found the current VAL-selected `ou_halflife` duration gate (`native.DURATION_THRESHOLDS`)
is overfit to validation and actively hurts OOS performance. This re-runs the entire
`total_notional_cap` and `asset_shares` sweeps from that doc with the gate disabled throughout
(`scripts/sweep_portfolio_concurrent_3asset_v4_20260712.py --duration-gate off`), to see whether
the earlier cap-tuning conclusions still hold on the corrected baseline.

## 1. total_notional_cap sweep, gate OFF vs gate ON (eth50/btc30/sol20 shares), oos_extended

| total_notional_cap | gate ON PnL | gate ON MDD | gate ON MTM MDD | gate OFF PnL | gate OFF MDD | gate OFF MTM MDD |
|---|---:|---:|---:|---:|---:|---:|
| uncapped | 69.70% | -38.21% | -45.01% | **188.95%** | **-29.10%** | **-36.25%** |
| 1.5 | 25.55% | -18.13% | -21.53% | **58.31%** | **-14.03%** | **-16.35%** |
| 2.0 | 34.86% | -23.54% | -27.79% | **84.24%** | **-16.52%** | **-20.71%** |
| 2.5 | 43.05% | -28.40% | -33.41% | **111.91%** | **-19.02%** | **-24.88%** |
| 3.0 | 54.42% | -32.45% | -38.26% | **139.72%** | **-21.00%** | **-27.51%** |
| 3.5 | 62.51% | -36.30% | -42.81% | **165.86%** | **-21.96%** | **-29.83%** |
| 4.0 | 66.52% | -37.81% | -44.74% | **174.07%** | **-24.39%** | **-31.55%** |

**The gate-off frontier strictly dominates the gate-on frontier at every single cap level** --
higher PnL *and* lower MDD simultaneously, not just a different point on the same trade-off curve.
The frontier shape itself (smooth, monotonic in the cap level) is preserved, so all the earlier
qualitative conclusions about `prealloc` behaving as a clean risk dial still hold -- only the
absolute numbers were being measured on a self-defeating baseline.

## 2. asset_shares sweep, gate OFF (total_notional_cap=3.0), oos_extended

| shares (eth/btc/sol) | gate ON PnL | gate ON MDD | gate OFF PnL | gate OFF MDD |
|---|---:|---:|---:|---:|
| 50/30/20 (user's requested order) | 54.42% | -32.45% | 139.72% | -21.00% |
| 40/35/25 | 49.70% | -31.15% | 118.90% | -22.87% |
| equal 33/33/33 | 48.30% | -29.29% | 109.10% | -26.29% |
| 60/25/15 (more ETH-heavy) | 54.41% | -33.61% | **154.15%** | **-18.11%** |

**The share ranking flips.** Under gate-on, equal-weighting had the best MDD and pushing ETH's
share past 50% only hurt (diminishing/negative returns). Under the corrected gate-off baseline,
**60/25/15 (more ETH-heavy than the user's requested 50/30/20) is now best on both PnL and MDD
simultaneously** -- strictly better than every other split tried, including 50/30/20 itself. This
means the "optimal" allocation is itself sensitive to the duration-gate decision; any share-tuning
done on the old gate-on baseline should be considered stale. This is presented as information,
not a push to change the user's stated priority order -- 50/30/20 (ETH>BTC>SOL) remains a
perfectly reasonable choice given it was an explicit preference, not a claimed optimum.

## Recommendation

Combining this with the earlier finding: **the current best-available configuration found in this
whole v1-v4 + sweep line of work is `cap_mode=prealloc`, duration gate OFF, `total_notional_cap`
picked from the section-1 frontier by risk tolerance** (e.g. 3.0 gives PnL +139.7%/MDD -21.0% on
oos_extended), with shares either the user's stated 50/30/20 or, if pure risk-adjusted return is
the goal, the more ETH-heavy 60/25/15 found here.

**Caveat that applies to the whole exercise, stated plainly:** this session has now iterated on
the same Jan-Jun 2026 OOS window many times (v1 uncapped -> v2 reject -> v3 scale -> v4 prealloc ->
cap sweep -> duration-gate test -> gate-off re-sweep), each time reading OOS results and adjusting
the next design choice. Every individual replay is fresh-forward/causal (no ledger leakage), but
repeatedly selecting configurations by looking at the *same* OOS window across this many rounds is
itself a familiar hazard in this project (same pattern flagged for Sigma6: "2026-03..06 now
heavily peeked"). Before treating any specific config above (cap level, shares, or the gate-off
decision itself) as a live candidate, it should be confirmed on a window that hasn't been used to
select anything in this session -- i.e. 2026-07+ data, not yet inspected here.

## Caveats

- Same modeling caveats as v1-v4 apply throughout (shared-ledger per-asset numbers are not
  dedicated-capital; new positions size off realized cash only; not a promotion artifact, no
  live wiring).
- The duration-gate-off decision itself was only tested as a full on/off toggle, not re-derived on
  a proper walk-forward selection scheme -- "off" beating the current VAL-selected thresholds does
  not prove "off" is optimal, only that the current thresholds are worse than no gate at all on
  this window.
