# Architecture-direction sweep (Sigma9/10/11) — all 4 proposed directions tested, NONE beat Sigma6

Status: `research_all_negative_not_adopted`

Last updated: 2026-07-06 KST

Lineage: user asked "is Sigma6 the best possible architecture?" Answer at the time was: strongest
result so far but with known fragility (lag-test sign flip, heavily-peeked OOS window). Four
concrete next directions were proposed, grounded in Sigma6's documented weaknesses:

1. **Multi-asset diversification** (chop-vulnerability of a single asset) → Sigma9
2. **Regime-specific expert models** (one generalist + post-hoc filter vs. specialists per regime) → Sigma10
3. **Learned/dynamic position sizing** (flat 3x/4x leverage vs. confidence/vol-scaled) → Sigma11
4. **Order-book microstructure features** (OHLCV-only info set) → blocked, not attempted

All three testable directions were implemented and evaluated on VAL (2025-07..12); none beat the
Sigma6 baseline (flat leverage, generalist signal, not_chop+stability filter: **+34.3%/-14.2%mdd
lev3, +71.1%/-15.9%mdd lev4**). None consumed the reserved OOS one-shot, per the pre-registered
"only spend OOS if VAL clears the bar" rule.

## Direction 1: Sigma9 — BTC+ETH 2-asset book (FAILED)

See `docs/model_contracts/sigma9_btc_eth_2asset_20260706_contract.md` for full detail. Only
BTC+ETH klines exist locally (no funding/OI/top-trader for BTC, so no BTC-specific Regime3 HMM).
BTC standalone (ungated) best VAL result: +16.6%. Blending 50/50 with ETH-Sigma6 REDUCED
risk-adjusted return in both leverage configs (return/MDD 2.42→2.22 at lev3, 4.47→3.37 at lev4).
**Verdict: not adopted.**

## Direction 2: Sigma10 — regime-specialist expert (UNSTABLE, FAILED robustness check)

`scripts/run_sigma10_regime_specialist_20260706.py`. Instead of training one generalist HGB on
all bars + gating trades post-hoc, train the ensemble ONLY on non-chop (`chop_prob < thr`) rows,
letting trees fit trend-regime statistics without chop-regime dilution. Same signal features,
same not_chop entry filter, same regime configs as Sigma6.

At the exact chop-training-threshold Sigma6 uses for its entry filter (0.42), the result looked
excellent: **lev3 +118.0% (mdd -19.2%, WR 55.6%), lev4 +107.8% (mdd -22.3%, WR 44.4%)** — a huge
apparent improvement over the +34.3%/+71.1% baseline, with gains spread across 4-5 months (not a
single-outlier month).

**But a sensitivity check across nearby thresholds immediately falsified it:**

| chop_train_thr | lev3 VAL | lev4 VAL |
|---|---|---|
| 0.34 | +26.1% (worse than baseline) | +39.0% (worse than baseline) |
| **0.42** | **+118.0%** | **+107.8%** |
| 0.55 | +81.3% | +44.4% (worse than baseline) |

The result swings wildly (+26% to +118%) across a modest hyperparameter range with no monotonic
trend — the exact signature of VAL noise/overfitting this project has seen before (Sigma8's
pruned-feature variant: VAL +34%→+51% while OOS collapsed to -12%). 0.42 is not special except
that it happens to equal Sigma6's already-tuned entry-filter threshold; there's no principled
reason the TRAINING-set filter should also be exactly 0.42, and the fact that it is the only value
in a 3-point sweep that clears the bar is a red flag, not a confirmation.

**Verdict: not adopted, OOS not spent.** The regime-specialist idea itself may not be wrong, but
this implementation's result is not distinguishable from noise given the evidence collected.

## Direction 3: Sigma11 — confidence/vol-scaled dynamic leverage (FAILED)

`scripts/run_sigma11_dynamic_leverage_20260706.py`. Per-trade leverage scaled by the model's own
`primary_quality_score` (confidence) and/or inverse ATR (vol targeting), frozen at entry, clipped
to `[1.5x, 2*lev_base]`. Compared against flat-leverage baselines at a MATCHED average realized
leverage (not raw PnL, since more leverage trivially means more return/risk).

| Variant | avg realized lev | VAL cost1 | MDD | return/MDD |
|---|---|---|---|---|
| flat lev=3 (baseline) | 3.00 | +34.3% | -14.2% | 2.42 |
| flat lev=4 (baseline) | 4.00 | +71.1% | -15.9% | 4.47 |
| dynamic lev_base=3, quality-only | 4.08 | +71.3% | -17.6% | 4.05 |
| dynamic lev_base=3, vol-target-only | 3.47 | +11.0% | -15.5% | 0.71 |
| dynamic lev_base=3, quality+vol | 4.27 | +36.8% | -17.6% | 2.09 |
| dynamic lev_base=4, quality-only | 5.44 | +44.7% | -30.1% | 1.48 |
| dynamic lev_base=4, vol-target-only | 4.27 | +63.0% | -17.2% | 3.66 |
| dynamic lev_base=4, quality+vol | 5.53 | +58.5% | -18.9% | 3.10 |

Every dynamic variant has a WORSE return/MDD ratio than the flat-leverage baseline at a comparable
average leverage. The best case (lev_base=3, quality-only) roughly matches flat lev4's raw PnL
(71.3% vs 71.1%) but at worse MDD (-17.6% vs -15.9%) — a wash at best, not an improvement.
Vol-targeting alone actively hurts (+11.0% vs +34-71% baselines) — likely because ATR is already
used as the stop/trail-distance unit, so scaling leverage by inverse-ATR partially cancels the
barrier's own vol-adaptivity in a way that isn't obviously beneficial.

**Verdict: not adopted, OOS not spent.**

## Direction 4: order-book microstructure — blocked, not attempted

`data/live/microstructure.duckdb` and `data/live/tail_risk.duckdb` are live-only accumulations
starting ~2026-05-03 (confirmed again 2026-07-06: files still only span the live period, no
2024-2025 backfill exists). With ~2 months of history there isn't enough data for a proper
train/val/OOS split. Revisit once several more months accrue (rough estimate: usable by
2026-Q4 at the earliest for a train split, later still for a clean OOS).

## Overall conclusion

All three testable "beyond Sigma6" architecture directions were implemented and none survived
even a VAL-only bar, let alone justified spending the reserved OOS look. This reinforces the
project's standing lesson ([[project-sigma8-sigma9-failed-attempts]] in memory): **Sigma6's edge
is structural** (1h cadence removes the cost-vs-signal problem; the not_chop+stability regime gate
removes the chop-bleed problem) and is not straightforwardly improved by more features, more
assets, regime-specific training, or fancier sizing — each of those either fails outright or
produces gains indistinguishable from noise. The one open, principled lever left is Direction 4
(order-book microstructure) once enough live history accrues; until then, **Sigma6 (lev3, steadier
WR 50%, or lev4, higher return) remains the recommended candidate**, with its known fragility
caveats unchanged.
