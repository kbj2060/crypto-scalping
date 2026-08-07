# Omega4.6.1 — Macro-Event Entry Veto Test (NULL RESULT: no trades affected)

Status: `research_null_result_not_applicable`

Last updated: 2026-07-06 KST

User request: test whether adding logic that "handles US major index releases well" improves
`omega4_6_1_duration_ou_halflife_risk_gate_20260630`'s extended Jan-Jun 2026 OOS result (see
`docs/model_contracts/omega4_6_1_extended_oos_20260706_retest.md`). Chosen design: veto new
entries in the immediate aftermath of major US macro releases (NFP, ISM manufacturing/services,
S&P Global flash/final PMI, FOMC decisions), then trade completely normally otherwise -- no
sizing boost, pure entry veto, reusing the exact rule-based event calendar already implemented
in `trading_bot_modules/omega5_live.py::Omega5LiveAdapter._macro_events_for_year()` (NFP = 1st
Friday 8:30am ET, ISM mfg/services = 1st/3rd business day 10am ET, flash PMI = on/after the 23rd
9:45am ET, FOMC = static verified 2026 dates). Script:
`scripts/test_omega4_6_1_macro_event_veto_20260706.py`.

## Result: 0 of 25 (and 0 of 33 pre-duration-gate) trades fall inside ANY tested veto window

Tested two window widths:

| Veto window | Trades affected (of 25 post-gate) | Trades affected (of 33 pre-gate) |
|---|---|---|
| User's chosen design: -30min / +15min | **0** | **0** |
| Omega5's original wider window: -30min / +120min | **0** | **0** |

Both windows produced an **identical, unchanged** result to the baseline (PnL +145.46%/MDD
-10.82%/25 trades post-gate; +141.14%/-13.78%/33 trades pre-gate) because literally none of the
model's 25-33 entries over the 6-month window happen to fall within 30 minutes before or 15-120
minutes after any of the ~40 relevant macro events in that period. Manual inspection of all 25
entry timestamps confirms they scatter across essentially random times of day/week with no
clustering near typical announcement times (13:30 UTC = NFP 8:30am ET, 15:00 UTC = ISM 10am ET,
19:00 UTC = FOMC 2pm ET) -- this is not surprising given the model trades roughly once per week
and the veto windows are narrow (45min-2.5h) relative to that frequency, so the a priori chance
of any overlap is low (order of 1-5% per trade even before accounting for the model's own
quality/regime filters).

## Follow-up: exposure haircut WHILE HOLDING through an event (not entry-based)

Entry-based veto is a null result because entries never land near events. But this model holds
positions a long time (avg ~99h, up to 282h) -- 11 of the 25 trades stay open through at least one
scheduled macro event during their hold, even though none of them ENTERED near one. This tests a
different countermeasure: temporarily haircut notional exposure during the event-window PORTION of
an already-open trade's hold (TP/SL/exit-head triggers unaffected -- they're raw price-move
barriers independent of notional per the model's contract), then restore full exposure after.
Approximated via bar-level log-return decomposition of each trade's price path (raw OHLC from
`training_features_2026_rebuilt.csv`), weighting event-window bars by the haircut scale before
re-aggregating -- ignores the transaction cost of dynamically resizing (a simplification).
Script: `scripts/test_omega4_6_1_macro_event_haircut_20260706.py`.

| Haircut during event bars | Trades touched | PnL | MDD | WR |
|---|---|---|---|---|
| none (baseline) | - | +145.46% | -10.82% | 0.520 |
| 50% notional | 11/25 | +151.20% | -10.82% | 0.520 |
| 25% notional | 11/25 | +153.79% | -10.82% | 0.560 |
| 0% (flat during event) | 11/25 | +156.20% | -10.82% | 0.560 |

A small, monotonic improvement (+5.7 to +10.7pp PnL as the haircut deepens), but **MDD does not
move at all** -- so this isn't really a risk-reduction win, just a mild return bump, and the
effect size is small relative to the approximations involved (log-return decomposition ignoring
resize transaction costs, only 11 trades' partial overlaps contributing, no out-of-sample
confirmation). **Not strong enough to recommend adopting** as-is, but directionally suggests price
action during scheduled-event windows was mildly unfavorable to this model's open positions in
this window -- worth a more rigorous test (larger sample, real resize cost accounting) if pursued
further, ideally on a model with more trades so the signal isn't resting on just 11 data points.

## Follow-up 2: lock in profit at T-30min before an event, let losers ride

User's third proposal: at T-30min before each scheduled event, if an open position is currently
in profit, force-close it now (lock the gain); if it's at a loss, leave it alone (let the model's
own TP/SL/exit-head keep running). Applied sequentially per trade across every event inside its
hold window -- the first T-30 checkpoint where the position shows unrealized profit force-closes
it (cost-consistent with `apply_max_hold_time_stop`'s reconstruction convention: entry price =
close at entry_timestamp, new exit cost = original fee/slip delta preserved). Script:
`scripts/test_omega4_6_1_macro_event_lock_profit_20260706.py`.

| | PnL | MDD | WR | trades force-closed early |
|---|---|---|---|---|
| baseline | +145.46% | -10.82% | 0.520 | - |
| **lock-profit rule** | **+148.23%** | -10.82% (unchanged) | **0.680** | 10/25 |

Win rate jumps sharply (52%→68%) because several trades that were HEADED toward a loss happened
to be briefly in profit at some T-30 checkpoint and got rescued into a small win (e.g. one trade
went from -7.43% to +6.49%, another from -5.26% to +5.02%). But this comes at a real cost: several
trades that were headed toward a BIG win got capped early (12.78%→4.70%, 11.37%→3.49%,
10.35%→0.26%) -- the rule forfeits upside on winners in exchange for rescuing some losers. Net PnL
effect is modestly positive (+2.77pp over 6 months) because the rescued-loser gains outweigh the
capped-winner losses, but MDD does not improve at all (no risk reduction, same drawdown profile).

**This runs directly counter to this project's own established "let winners run" principle**
(the trailing-stop/no-early-exit design is exactly what turned Sigma5's failure into Sigma6's OOS
pass -- see [[project-sigma6-regime-trend-best]]). Capping winners early at 12.78%→4.70% etc. is
the same mistake pattern that hurt earlier scalping-style Alpha/Omega variants. The higher win
rate is real and could matter for psychological/operational consistency, but the PPL improvement
is small and comes with a philosophy trade-off the project has previously found costly elsewhere.

## Conclusion

Three variants tested, all informative but none strong enough to promote outright:

1. **Pure entry veto**: null result -- untestable on this model (entries never land near events
   given ~1 trade/week frequency).
2. **Exposure haircut while holding through an event**: small, MDD-neutral PnL gain (+5.7 to
   +10.7pp depending on haircut depth), based on only 11 trades' partial overlaps -- directionally
   interesting, too weak/approximate to promote.
3. **Lock profit at T-30min, let losers ride**: WR jumps 52%→68%, PnL +2.77pp, MDD unchanged --
   but caps several of the biggest winners early, in tension with the project's core "let winners
   run" lesson from Sigma6.

If pursuing the "handle macro releases well" hypothesis further, it would be more productive on a
higher-frequency candidate (Sigma6/Sigma7-family, ~1 trade per 4-5 days) with more data points to
support any of these three designs with real statistical confidence.
