# BTC Sigma3 1h-Entry + 5m-Exit-Precision — CLOSED 2026-08-05

## What was tried

User asked: can 30m/1h decide entry+sizing while 5m learns/handles the exit? Prior
memory search surfaced two directly relevant findings before any new code was
written:

1. `project-sigma3-1h-breakthrough` (2026-07-05): BTC Sigma3 (1h bars + trend-
   scanning + HGB) is the strongest fresh-window BTC result in project history
   (OOS cost1 +7.34%, TP-hit-rate 40% vs 5m's ~9%), but failed its pre-registered
   cost3 gate by a small margin (OOS cost3 -3.88%, near breakeven). Critically, a
   companion experiment (Sigma4: 5m-native decision WITH 1h context) passed VAL
   but failed catastrophically on one-shot OOS -- the memory's explicit conclusion
   was "the 1h DECISION CADENCE itself is the working lever... let the 1h model
   own the signal and only allow 5m EXECUTION TIMING, do NOT train a 5m-native
   decision model."
2. Inspecting `scripts/replay_omega6_v2_variants_20260704.py::run_variant`
   (Sigma3's own replay engine) found it checks TP/SL against the 1h bar's
   **close price only, once per hour** -- not intrabar high/low. This is coarser
   than every other barrier-touch check used elsewhere in this project (which use
   high/low touches, e.g. `_reason_and_return`), and is a plausible mechanical
   reason Sigma3 was giving back gains / missing TP touches before its own
   1h-close-based check registered them.

This motivated the CHEAPEST possible version of the user's idea before any
learned-model investment: hold Sigma3's exact frozen 1h entry decisions fixed
(same tape, same `qt0.7/p0/tp1.5/sl1.0` config) and change ONLY exit resolution
from 1h-close-only to a 5m intrabar high/low walk-forward (matching the project's
standard barrier convention). `scripts/eval_btc_sigma3_5m_exit_precision_20260805.py`.

## Result

**VAL (2025-07..12) and the already-spent OOS window (2026-03-02..06-30) both
looked like a breakthrough**: cost3 PnL flipped from Sigma3's original failing
-3.88% (OOS) to **+4.71%** with the 5m exit resolution (VAL: +3.47% -> +1.59%,
still positive). Trade count roughly doubled (77->191 OOS) because faster exit
resolution frees up "slots" for more of the tape's already-present entry signals
to fire (same slot-occupancy mechanism documented for ETH in
[[project-eth-omega461-slot-occupancy-trade-count-20260728]]), and time-stop exits
nearly vanished (barriers are reached far more often when checked every 5 minutes
instead of once an hour).

**But 2026-03-02..06-30 was already consumed by Sigma3's original one-shot look**
(explicitly flagged as spent in `docs/model_contracts/sigma3_1h_trendscan_20260705_contract.md`).
Per that same doc's own recommendation ("reserve a new untouched window, e.g.
2026-07+"), the identical test was re-run on the genuinely fresh
**2026-07-01..07-20** slice (never scored by any prior run):

| | 1h-close baseline | 5m intrabar exit |
|---|---:|---:|
| cost1 PnL | +1.85% | **-8.30%** |
| cost3 PnL | -0.29% | **-10.67%** |
| trades | 12 | 24 |
| win rate | 41.7% | **8.3%** |

**The improvement completely reverses on genuinely fresh data.** Win rate
collapses from 41.7% to 8.3% (2/24 wins). This is the exact "wins on
VAL/already-observed-OOS, fails on truly fresh data" signature that has closed
every other line in this session's arc (JEPA deep features, TP-first meta-label,
barrier/horizon calibration) and much of the project's broader history.

## Verdict: CLOSED (mechanical version)

The cheap, purely-mechanical version of "1h entry + 5m exit" — just checking
barriers more precisely, no learned model — does not hold up. Two plausible
(non-exclusive) explanations, neither investigated further given the small fresh
sample (12/24 trades, 19 days):
1. The 5m-resolution mechanism's apparent OOS(2026-03..06) win was itself
   already partly a product of having been looked at (selection risk on a
   "spent" window, even though no parameter was tuned against it) rather than a
   real, stable effect.
2. The "more trades fill freed-up slots" mechanism is double-edged: it also
   admits more low-conviction entries the coarser baseline's slower turnover
   was accidentally filtering out, which could show up as thin-sample noise in
   either direction on a short fresh window.

## What was NOT tried

The user's original, more ambitious idea -- a *learned* 5m exit-timing model
(not just finer mechanical barrier checking) -- was not built. Given: (a) this
cheap mechanical precursor already failed the fresh-window test, (b) ETH's
[[project-eth-omega461-exit-logic-experiments-20260721]] already ran 21+ rounds
of learned/heuristic exit-timing mechanisms on a WORKING baseline with far more
data and infrastructure, all failing OOS, and (c) BTC has no working baseline to
begin with (every entry-side attempt this session closed) -- a learned 5m exit
model is a materially larger investment with a weak prior for BTC specifically.
Not recommended as the next move without a different signal first.

## Artifacts

- `scripts/eval_btc_sigma3_5m_exit_precision_20260805.py`
