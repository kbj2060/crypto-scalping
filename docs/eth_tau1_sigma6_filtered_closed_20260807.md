# ETH Tau1 (Sigma6-filtered second leg) — CLOSED 2026-08-07

## What Tau1 was

Tau1 named a combination layer, not a new signal model: live Omega4.6.1 (real, executing ETH
capital) plus a second, independent-margin "Sigma6-filtered" leg (a regime-gated Sigma3-1h
HistGradientBoosting direction model, quality_thr=0.60, not_chop regime gate reg_thr=0.50,
lev=3.0/sl=1.5 ATR/trail=5.0 ATR/min_profit=2.0 ATR/max_hold=144h/cooldown=3h), combined with a
`regime_tiebreak` conflict-resolution rule. It ran as a paper-only live shadow
(`scripts/live_sigma6_regime_tiebreak_shadow_20260801.py`, systemd unit `tau1-shadow.service`,
`order_submission_supported: false`) from 2026-08-01, tracking a hypothetical Sigma6-filtered
position and a hypothetical regime_tiebreak-weighted combination against the real live Omega4.6.1
ETH position. No live capital was ever allocated to it.

## Why it's closed

**1. The live-persisted model was trained with no holdout, in contradiction with its own docstring.**
`scripts/train_sigma3_1h_hgb_persist_20260801.py` originally fit on the full concatenated
2024-01-01..2026-07-20 dataset with no `train_mask` at all, despite claiming to reproduce
`train_sigma3_1h_hgb_20260705.py`'s `TRAIN_END=2025-06-30` holdout — meaning the deployed model had
directly trained on labels from the exact VAL_2025Q4/OOS_2026H1 windows used to justify running Tau1
live at all. Fixed 2026-08-07 (added the matching `TRAIN_END` mask, retrained, `train_rows=13127`
confirmed to match the original research script's `report_seedA.json`).

**2. Fixing the holdout revealed the deployed single-seed model had no real edge.** The headline
backtest numbers quoted when Tau1 was proposed (VAL_2025Q4 Sigma6-filtered +15.40%/Tau1 combined
+90.70%; OOS_2026H1 Sigma6-filtered +31.47%/Tau1 combined +263.48%) came from a 2-seed
(270705+270710) averaged ensemble tape. The actual live-deployed artifact is only ONE of those two
seeds (270705). Re-measured with the corrected holdout: OOS Sigma6-filtered leg alone flips to
**-30.18%** (well-powered, 69 trades), Tau1 combined OOS drops to +108.01% with MDD nearly doubling
(-19.61% Omega-alone -> -30.51% combined) — Sigma6-filtered was a net drag on the live shadow, not a
contributor.

**3. Formal seed-diversity gate (N=8 genuinely random seeds) confirms the original number was noise.**
Per this project's Seed-Diversity Ensemble Promotion Gate (CLAUDE.md), retrained with 8 independently
random seeds (`np.random.default_rng(20260807).integers(1,999_999,size=8)` ->
`[120758, 227310, 237374, 266963, 449908, 543934, 681691, 804367]`, same TRAIN_END holdout/
hyperparameters). Results (VAL / OOS pnl,mdd per seed):

| seed | VAL pnl/mdd | OOS pnl/mdd |
|---|---|---|
| 120758 | +5.77%/-24.22% | +2.76%/-34.15% |
| 227310 | +22.78%/-26.06% | -3.15%/-36.64% |
| 237374 | +24.43%/-25.98% | -17.12%/-49.28% |
| 266963 | -23.66%/-45.63% | -25.07%/-52.70% |
| 449908 | -30.50%/-43.51% | -24.86%/-51.62% |
| 543934 | +10.54%/-28.40% | -18.18%/-42.84% |
| 681691 | +39.91%/-23.24% | -12.64%/-42.98% |
| 804367 | -1.08%/-32.76% | -24.25%/-46.95% |

OOS positive: **1/8**. VAL positive: 5/8 (VAL itself flips sign across seeds, -30.5%..+39.9%). The
8-seed averaged ensemble is *also* OOS-negative (+12.21% VAL / **-8.83% OOS**, MDD -45.78%). This
fails the seed-diversity gate outright — the original +31.47%/+263.48% OOS numbers were pure
seed-variance noise from an unrepresentative 2-seed clustered draw, not a reproducible signal.

Full per-seed data: `tmp/eth_tau1_sigma3_1h_seed_diversity_20260807/{report.json,
per_seed_results.csv}`.

## Decision

**Sigma6-filtered/Tau1 as a second ETH sleeve is closed.** No further seed sweeps, threshold
re-tuning, or regime-gate variants should be attempted on this same trend-scanning 1h feature/label
setup — the seed-diversity result shows there is no real, reproducible edge to tune. A future attempt
would need a materially different feature set, label, or architecture, not more search on this one.

**ETH live capital remains Omega4.6.1 alone** (unchanged — Tau1 never held live capital).

## What was decommissioned (2026-08-07)

- `train_sigma3_1h_hgb_persist_20260801.py`: holdout bug fixed (kept in repo for reference/history,
  not because the model is still in use).
- Live shadow loop (`scripts/live_sigma6_regime_tiebreak_shadow_20260801.py`, systemd unit
  `tau1-shadow.service`) stopped/disabled.
- Dashboard: removed the "Tau1 새도우" panel (`dashboard/live/index.html`,
  `dashboard/live/app.js` — `refreshSigma6TiebreakShadow`/`renderSigma6TiebreakShadow`/
  `renderSigma6TradeJournal` and related helpers), the `/api/sigma6-tiebreak-shadow` endpoint and its
  `tau1_shadow` ops-status supervisor entry (`dashboard/server.py`), and the associated CSS
  (`dashboard/live/styles.css`).
- Historical shadow data (`data/live/sigma6_regime_tiebreak_shadow/{state.json, equity_curve.jsonl,
  closed_trades.jsonl}`) left in place as a historical record, not deleted.

## References

Full technical detail and lookahead/contamination audit trail: see the auto-memory entry
`project-eth-tau1-shadow-holdout-fix-reveals-negative-single-seed-oos-20260807.md` and
`project-tau1-name-spec-20260801.md` (original architecture spec, mechanics description still
accurate — only the performance claims are now known to be seed-variance artifacts).
