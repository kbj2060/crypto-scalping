# Sigma4 — 5m Decisions + 1h Context (Multi-Timeframe)

Status: `research_failed_one_shot_5m_cadence_does_not_generalize`

Last updated: 2026-07-05 KST

Lineage: built per user request (2026-07-05): (a) cost3 (3x-fee stress) is too strict — gate on
cost1 (realistic 1x fees); (b) make trading decisions every 5m, using 1h only as reference.

## Design

5m decision cadence with 1h trend context as backward-looking reference features.
- 5m features (38, from `compute_features` at 5m) + 8 CAUSAL 1h-context features (1h momentum
  1/3/6/12, 1h RSI, 1h realized-vol, 1h past-bar OLS slope + sign), merged via `merge_asof` on the
  1h bar's completion time (H+1:00) so no lookahead. The 1h trend-scanning *label* is forward-
  looking and deliberately NOT used as a feature.
- 5m trend-scanning labels (windows 12-96 bars = 1h-8h forward, threshold 2.5).
- 5-seed HistGradientBoosting ensemble, train 2024-01..2025-06. 5m barriers (tp 2.5-5x / sl
  1.5-2x ATR, atr median 0.22%, max_hold 288 bars = 24h, cooldown 6-12 bars).
- **Relaxed gate (per user): cost1 PnL > 0, cost1 MDD >= -20%, cost3 MDD >= -25%, trades >= 60,
  months >= 5.** cost3 PnL reported but NOT required positive.
- Scripts: `build_5m_mtf_dataset_20260705.py`, `run_sigma4_5m_mtf_20260705.py`.

## Result: validation passed (relaxed), one-shot FAILED decisively

**Validation (2025-07..12): 2/54 pass.** Best `qt0.75/p0/tp3.5/sl1.5/cd6`: cost1 +7.58%
(MDD -7.18%), 60 trades, WR 40%, cost3 -4.23%. But monthly cost1 was weak/back-loaded: Jul -2.42,
Aug +2.32, Sep -2.29, Oct -1.19, Nov +3.88, Dec +7.38 — only 3/6 months positive, profit
concentrated in the final two months (flagged before the one-shot).

**One-shot 2026-03-02..06-30 (frozen `qt0.75/p0/tp3.5/sl1.5/cd6`):**
- cost1: **-20.48%** (MDD -22.02%), 138 trades, **WR 24.6%**, **all 4 months negative**
  (Mar -4.72, Apr -4.40, May -6.77, Jun -6.59)
- cost3: -45.33%

The validation "pass" (concentrated in Nov-Dec 2025) did not generalize at all; out-of-sample
win rate collapsed from 40% to 24.6%, trade count rose to 138 (overtrading), and every OOS month
lost. (Caveat noted: 2026-03..06 was scored once before for Sigma3-1h, so this is its 2nd use;
but the failure is so large this doesn't affect the conclusion.)

## Conclusion: decisively confirms the frequency finding

Reverting to a 5m decision cadence — even with 1h context features AND the relaxed cost1-only
gate — reintroduced exactly the overtrading/noise-domination problem that moving to 1h had
solved. The 5m signal's win rate is not stable out-of-sample.

| | Sigma3 **1h** decisions | Sigma4 **5m** decisions (+1h context) |
|---|---|---|
| OOS cost1 | **+7.34%** | -20.48% |
| OOS win rate | 47% | 24.6% |
| OOS months positive | 3/4 | 0/4 |

**The working lever is the 1h decision cadence itself, not features or cost-gate relaxation.**
The 1h context helped in-sample but could not rescue 5m decisions out-of-sample. Sigma3-1h
(1h decisions, cost1 +7.34%/MDD -15% on the fresh window) remains the only approach whose signal
survived onto genuinely untouched data, and is the recommended live candidate if evaluated at
realistic (cost1) fees. If the operational requirement is truly to act every 5m, the viable form
is to let the 1h model own the signal and merely allow 5m execution timing of its entries/exits
— NOT to train a 5m-native decision model, which does not generalize here.
