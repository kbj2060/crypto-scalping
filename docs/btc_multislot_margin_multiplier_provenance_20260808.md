# BTC multi-slot margin multiplier — provenance correction (2026-08-08)

**The recorded adoption number was wrong, and correcting it also invalidates the SELECTION.
No live change has been made; the decision is escalated.**

Script: `scripts/resweep_btc_multislot_margin_multiplier_fullreplay_20260808.py`
Results: `tmp/btc_multislot_margin_resweep_20260808/results.json`

## The bug

The 1.5x margin multiplier was adopted on 2026-08-07 by a sweep run **on the N=3 gated ledgers**,
i.e. by rescaling trade returns per multiplier. That is invalid for this stack: `margin_fraction`
is an **input to the exit head** (notional / leverage / exposure sit in `pos_values`), so changing
the multiplier changes the exits and therefore the ledger itself. A rescaled ledger answers "what
if the same trades were bigger", which is not the question. The live loop applies the multiplier
**before** the exit head, so only a full causal replay matches live.

No script for the original sweep was ever committed — there was no reproducible path at all. There
is one now.

## Corrected numbers (full causal replay, N=3, cost x3, exit threshold 0.95)

| m | VAL gated PnL / MDD | OOS gated PnL / MDD | OOS Q1 / Q2 / Q3 |
|---|---|---|---|
| 1.0 | +22.32 / −3.39 | +16.72 / −7.27 | 9.86 / 6.90 / −0.61 |
| 1.25 | +28.45 / −4.23 | +21.00 / −9.03 | 12.29 / 8.59 / −0.77 |
| **1.5 (live)** | +34.80 / −5.05 | **+25.30 / −10.77** | 14.69 / 10.27 / −0.92 |
| 1.75 | +41.39 / −5.87 | +33.50 / −12.48 | 17.07 / 15.28 / −1.08 |
| 2.0 | +48.21 / −6.69 | +38.56 / −14.17 | 19.43 / 17.48 / −1.25 |

Equivalence check passed: the m=1.5 cell reproduces the already-published full-replay figures
(+25.30 / −10.77) to the decimal, which also proves the `prepare()` extraction shared with
`eval_btc_multislot_shadow_with_regime_sizing_20260808.py` is behaviour-preserving.

## Three findings, in order of severity

**1. The recorded adoption figure was wrong.** Recorded +19.98% / −10.40%; actual
+25.30% / −10.77%. PnL off by 5.3pp. The error direction is favourable, which is why nothing
looked wrong downstream.

**Why it went unnoticed:** the rescale happens to match the replay at 1.25x (+21.0 / −9.0 recorded
vs +21.00 / −9.03 replayed) — at that size the exit-head decisions do not change. It only diverges
at 1.5x, where the larger exposure moves exits. The original note's reassurance that "OOS peak was
actually 1.25x, rule held, no OOS cherry-pick" rested on a **false shape**: under correct replay
PnL is monotone increasing and 1.25x is not a peak.

**2. The stated VAL rule is degenerate, and correctly applied it does not select 1.5x.**
VAL PnL is monotone increasing in m, and VAL MDD never reaches the −8% bar (worst −6.69 at 2.0x).
So "highest VAL PnL subject to VAL MDD ≥ −8%" reduces to **"take the largest multiplier in the
grid."** Over the original grid `{1.25, 1.5, 1.75, 2.0}` (recorded in the adoption memory, so this
is not an artifact of the reconstructed grid) the rule selects **2.0x, not 1.5x**.

**3. The rule's own OOS gate then rejects its own selection.** 2.0x scores OOS PnL +38.56
(bar +19.7 ✓), MDD −14.17 (bar −12.4 ✗), worst quarter −1.25 (bar −4.0 ✓). The pre-registered
procedure says "otherwise the axis is closed (no re-tuning on OOS)". **A correct execution of the
stated rule therefore rejects the multiplier extension entirely and falls back to 1.0x.**

## What this does and does not mean

It does **not** mean 1.5x is a bad operating point. On its own numbers 1.5x passes all three OOS
gates (+25.30 ≥ +19.7, −10.77 ≥ −12.4, −0.92 ≥ −4.0), and it is the largest multiplier whose OOS
MDD clears the gate — the crossing sits between 1.5 and 1.75.

But **choosing 1.5x for that reason is OOS cherry-picking**, which is exactly what the rule was
written to prevent. The honest statement of 1.5x's provenance is now:

> 1.5x is not VAL-selected. It is the multiplier that passes the OOS gates, chosen after seeing
> the OOS numbers.

That is a materially weaker claim than the one on record.

## No live change made

`BTC_MULTISLOT_SHADOW_MARGIN_MULT` is left at 1.5 pending an explicit decision. Changing live
sizing is not a bookkeeping fix, and the corrected sweep read a window that has already been spent
three times (promotion, multislot gate, multiplier) — acting on it is itself a form of re-tuning.

Options, with the tradeoff stated:

- **(a) Revert to 1.0x** — strict rule compliance. The extension is rejected as the rule dictates.
  Costs the measured OOS PnL difference (+25.30 → +16.72) and gains MDD (−10.77 → −7.27).
  Resets the shadow's live record, including the czz_trend overlay shadow that was configured at 1.5.
- **(b) Keep 1.5x, restate provenance** — no behavioural change; the record says plainly that 1.5x
  is OOS-gate-derived, not VAL-selected. Carries the weaker claim honestly.
- **(c) Let the live shadow record decide** — it was always designated the binding referee, and it
  is unaffected by which backtest number is on file.

(b) + (c) is the recommendation: the shadow exists precisely because backtest selection on a
thrice-read window is not trustworthy, and reverting now would destroy the only evidence stream
that is not contaminated by that problem. (a) is the right answer if strict rule compliance is the
priority.

## Reusable lesson

**Never evaluate a sizing change by rescaling an existing ledger when sizing feeds the exit model
— re-run the replay.** And when a selection rule's constraint never binds inside the tested grid,
the rule is not selecting on data; it is selecting on grid extent. Check that the constraint
actually bites before trusting the pick.
