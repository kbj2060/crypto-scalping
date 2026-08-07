# Sigma3 — 1h Bars + Trend-Scanning + Gradient Boosting

Status: `research_closest_yet_but_cost3_one_shot_fail`

Last updated: 2026-07-05 KST

Lineage: third from-scratch approach after [[sigma1]]/[[sigma2]]. Changes THREE things at once
vs all prior attempts, each tied to a specific documented failure diagnosis.

## Design

| Change | vs prior | Rationale |
|---|---|---|
| **1h time bars** (resampled from clean 5m) | 5m | 5 prior approaches all "cost1+/cost3-": edge < 3x cost at 5m. At 1h a trade spans a 2-4% move so 0.42% cost3 round-trip is a small fraction |
| **Trend-scanning labels** (Lopez de Prado, max-\|t\|-value forward linear fit, windows 3-48h, threshold 2.5) | zigzag / barrier | Web research (2026-07): AEDL is SOTA but too fragile to reimplement; trend-scanning is robust, adapts horizon per bar, and its t-stat naturally filters to significant moves |
| **HistGradientBoosting 3-class** (2 seeds → 5-seed ensemble) | neural TabM/GRU | Every neural attempt overfit instantly (holdout loss min at epoch 1). Trees can't memorize sequences; multi-horizon features (returns over 1-24 bars) carry the temporal context |

- 38 stationary features at 1h (multi-horizon returns/vol, RSI/MACD/BB, ATR%, BTC + ETH-BTC
  spread, funding/OI z-scores, taker imbalance, candle shape, rolling skew/kurt, calendar).
- Train 2024-01..2025-06 (18mo, ~13k bars). Validation 2025-07..12 (6mo, never used for
  selection before). One-shot 2026-03-02..06-30 (never touched by any model/search).
- Order-book/execution duckdb data (`data/live/microstructure.duckdb`) inspected: OBI, taker
  flow, spoofing/absorption scores, L2 depth 20 levels, liquidation flow — but only spans
  **2026-05-03..07-05 (~2 months, live-only)** and does not exist for 2024-2025, so it CANNOT be
  a historical training feature. Reserved for future live use.
- Scripts: `build_1h_trendscan_dataset_20260705.py`, `train_sigma3_1h_hgb_20260705.py`,
  `train_sigma3_1h_ensemble_20260705.py`, `replay_sigma3_1h_gates_20260705.py`.

## Results

**Initial 2-seed sweep (27 configs/seed): 0/27 joint pass — severe seed instability.** seedB
alone passed several configs cleanly (e.g. qt0.7/p2/tp1.5/sl1.0: cost1 +42.2%/cost3 **+26.3%**/
MDD -7-9%/WR 52%) but seedA was NEGATIVE at the same configs (cost1 -4.2%) — the two seeds
produced fundamentally different signal distributions (seedA early-stopped at 143 iters, seedB
ran 400). The pre-registered 2-seed sign-consistency gate correctly rejected this. Unlike the 5m
models this was NOT a "cost3 always negative" wall — seedB proved cost3-positive-with-good-MDD is
reachable at 1h; the problem was pure variance.

**5-seed ensemble (fixed 250 iters, variance-reduced, FINAL validation look): 1/27 gate pass.**
`qt0.7/p0/tp1.5/sl1.0`: val cost1 +13.75% (MDD -8.64%), cost3 +3.47% (MDD -14.51%), 90 trades,
WR 45.6%, 6 months. Monthly (cost1): Jul -0.21, Aug +0.70, Sep +2.71, **Oct +13.66**, Nov +1.48,
Dec -4.08 — 4/6 positive but profit concentrated in October (flagged).

**One-shot on untouched 2026-03-02..06-30 (frozen `qt0.7/p0/tp1.5/sl1.0`):**
- cost1: **+7.34%** (MDD -15.02%), 78 trades, WR 47.4%, **3/4 months positive** (Mar +6.84,
  Apr +2.05, May +1.70, Jun -2.54)
- cost3: **-3.88%** (MDD -20.43%) → **fails the cost3>0 gate. OOS_PASS = False.**
- Exit reasons: take-profit 31/78 (**40%**) vs stop-loss 37, time-stop 9.

## Verdict: FAILED the pre-registered one-shot (cost3<0), but a qualitative breakthrough

This is by far the strongest fresh-window result in the project's history, and the ONLY approach
whose signal survived onto genuinely untouched data with the direction intact:

| Metric | 5m approaches (Sigma1/2, barrier-relabel, m7-free) | Sigma3 1h |
|---|---|---|
| Fresh/OOS cost3 | -20% to -70% collapse | **-3.88%** (near breakeven) |
| Take-profit hit rate | ~9% | **40%** |
| OOS cost1 sign | flipped negative | **+7.34%, 3/4 months positive** |

The frequency change did exactly what the diagnosis predicted: barriers are now reachable (TP
40% vs 9%), and the cost3 gap closed from >20 points to ~4 points. At realistic 1x costs
(cost1) the frozen config is +7.34%/-15% MDD over 4 fresh months — tradeable if execution is near
cost1; the failure is specifically under the conservative 3x-cost stress.

Per one-shot discipline, the 2026-03-02..06-30 window is now CONSUMED for this frozen config; no
variant fishing was done after seeing the OOS number. The result stands as a fail on the
pre-registered gate.

## Recommended next steps (not executed — would need a fresh OOS window or more data)

1. **Push frequency further / information-driven bars**: 2h/4h time bars, or dollar/volume bars
   (crypto-specific, research-recommended) — should widen the cost3 margin further.
2. **Raise win rate at entry**: the edge is directional but entries are near coin-flip (WR
   47%); a meta-labeling second stage (Lopez de Prado) to filter low-conviction entries could
   lift cost3 over the line without touching the primary signal.
3. **Once ~6+ months of order-book history accumulates** (currently only 2 months live), add the
   microstructure features (OBI, absorption, liquidation flow) — genuinely new information not
   in the OHLCV universe that 5+ approaches have now exhausted.
4. Reserve a NEW untouched window (e.g. 2026-07+ as it accumulates) for the next one-shot; the
   2026-03..06 window is spent for this model family.

The Omega6 v2 frozen winner (m7-dependent, unreproducible on new data) remains the only config to
have passed both validation and a one-shot in project history; Sigma3 is the closest challenger
and the only one built on a reproducible, m7-free, going-forward-servable feature set.

## Follow-up levers tested (2026-07-05, both FAILED to improve on validation)

Two of the recommended next steps above were executed and evaluated on the SAME validation
window (2025-07..12), so no additional one-shot was spent:

**(a) Lower frequency 2h / 4h** (`scripts/run_sigma_freq_experiment_20260705.py`, same
recipe/grid, ATR barriers auto-scale): both WORSE than 1h. 2h best cost3 -6.93% (0/27 pass);
4h best cost3 -7.56% (0/27 pass), vs 1h ensemble +3.47%. Root cause: training samples collapse
(1h 13,127 → 2h 6,564 → 4h 3,282 bars), starving the model faster than the bigger-move benefit
helps. **1h is the frequency sweet spot** among {1h,2h,4h}; going lower is net-negative here.

**(b) Meta-labeling** (`scripts/run_sigma3_metalabel_20260705.py`, Lopez de Prado: 2nd HGB
predicts P(this primary signal wins) from the 38 market features, filters entries): 0/16 pass.
Best (base0.5/meta0.65/p2): cost1 +6.86%, cost3 **-1.19%**, MDD -12.4%, but only 39 trades
(< 40 floor). Meta-filtering made validation cost3 WORSE than the plain 1h ensemble's +3.47% at
qt0.7 — i.e. a simple high primary-confidence threshold (qt0.7) is a better, cleaner filter than
the learned meta-model, whose win-probability signal did not transfer out-of-sample. Meta
training-set win rate was 62.5%, but that in-sample edge (partly primary overfit) didn't
generalize.

**Net:** the plain 1h Sigma3 ensemble at qt0.7/p0/tp1.5/sl1.0 remains the champion of this
family. Neither frequency-lowering nor meta-labeling beat it. The genuinely untried, higher-
potential levers remain: dollar/volume (information-driven) bars instead of time bars, and the
order-book microstructure features once ~6+ months of history accumulate (currently 2 months
live-only). Both should be tested against a FRESH untouched window (2026-07+), not 2026-03..06
(spent).
