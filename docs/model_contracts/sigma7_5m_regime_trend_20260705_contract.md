# Sigma7 — 5m Decisions + 1h/6h Context + Regime Filter + Trailing (FAILED OOS)

Status: `research_failed_oos_5m_cadence_not_viable_even_with_all_winning_pieces`

Last updated: 2026-07-05 KST

Lineage: built per user request (2026-07-05) — 5m decision cadence with 1h + 6h as auxiliary.
Combines every technique that worked elsewhere: 5m multi-timeframe features + Sigma6 regime
filter (not_chop + stability) + Sigma5 let-winners-run trailing trend-following.

## Design

- 5m base features (38) + 1h context (7) + 6h context (7) = 52 features; all higher-TF context
  backward-looking, merged causally on completion time. 5m trend-scanning labels.
- 5-seed HGB ensemble, train 2024-01..2025-06. 5m tape + 5m-native Regime3 wide24 (bull/bear/chop)
  + CryptoMamba stability merged.
- Backtest = Sigma6 (trailing stop + not_chop/stability regime filter) reused verbatim, with
  5m-appropriate barriers (trail 8-20xATR, sl 4-8xATR, max_hold 576-1152 bars = 2-4 days).
- Scripts: `build_5m_mtf2_dataset_20260705.py`, `run_sigma7_5m_regime_trend_20260705.py`.

## Result: strong validation, decisive OOS failure — the regime filter does NOT rescue 5m

| Config | VAL cost1 | **OOS cost1** | OOS WR | OOS no-filter control |
|---|---|---|---|---|
| A lev3/trail20/sl4/mh1152 | +141.1% | **-40.3%** | 18.9% | -44.3% |
| B lev3/trail20/sl8/mh1152 | +84.3% | **-42.3%** | 28.6% | **-4.7%** (filter HURT) |
| C lev3/trail12/sl4/mh576 | +55.3% | -0.6% | 30.4% | -1.5% |

OOS = 2026-03..06. Validation reached +141% but every config collapsed OOS; win rate fell to
~19-30%. Critically, the **no-regime-filter control was similar or BETTER** (config B: -42.3% with
filter vs -4.7% without) — the regime filter that rescued 1h trend-following (Sigma6 +45.9% OOS)
does NOT help at 5m, and can hurt. This shows the 5m core signal is noise-dominated: the regime
isn't the problem, the 5m entry timing is, and no amount of MTF context / regime filter / trailing
stop fixes it.

## Verdict: 5m decision cadence is not viable in this project (2nd confirmation)

This is the second thorough 5m-cadence failure (after Sigma4 OOS -20%), and this time WITH every
winning technique bolted on (1h+6h context, regime filter, trailing trend-follow). The same
recipe at 1h (Sigma6) gave OOS +45.9%; at 5m it gives -40%. The conclusion is now strongly
evidenced: **the working decision cadence is 1h; 5m entries do not carry a generalizing edge.**

**Recommended architecture for 5m responsiveness without a 5m decision model:** let the 1h model
(Sigma6) OWN the entry/exit decision, and use 5m only for EXECUTION TIMING — when Sigma6 fires
"enter long", capture a favorable fill on the 5m grid within the hour instead of waiting for the
next 1h open, and likewise time exits on 5m. This keeps the 1h signal's real edge (OOS +45.9% /
~+30% latency-adjusted) while giving 5m-level execution precision. It does NOT retrain a 5m
decision model, which repeatedly fails to generalize here.

Sigma6 (1h decisions + regime filter + trailing) remains the recommended live candidate.
