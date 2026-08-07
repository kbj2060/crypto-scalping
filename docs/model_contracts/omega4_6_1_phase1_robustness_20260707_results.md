# Omega4.6.1 — Phase 1 Robustness Audit Results (2026-07-07)

Runs `scripts/audit_omega4_6_1_phase1_robustness_20260707.py` per Phase 1 of
`omega4_6_1_improvement_roadmap_20260707.md`. Diagnostic only -- no live wiring change, no new
model. Frozen config: h48qual+zig075 greedy router, static TP/SL, duration gate (VAL-selected
threshold 0.005417).

## 1. Cost stress test (cost_mult 1x/2x/3x)

| window | cost1x | cost2x | cost3x |
|---|---|---|---|
| VAL | +54.88% / mdd -31.11% / n=22 / wr 0.455 | +24.37% / mdd -32.00% / n=22 / wr 0.409 | +19.57% / mdd -33.14% / n=22 / wr 0.409 |
| OOS | +145.34% / mdd -10.13% / n=24 / wr 0.542 | +112.80% / mdd -16.19% / n=25 / wr 0.520 | +104.56% / mdd -16.66% / n=25 / wr 0.520 |

**Passes the Alpha1 cost-robustness bar.** PnL decays under higher cost assumptions (expected --
more fee/slippage) but stays clearly positive at 3x cost in both windows (VAL +19.57%, OOS
+104.56%). MDD degrades moderately (VAL -31%->-33%, OOS -10%->-17%) but does not blow up. This is
NOT the "wins at cost1, dies at cost3" pattern this project has flagged elsewhere as a fake edge --
first time this check has actually been run for Omega4.6.1.

## 2. Leave-one-out (jackknife) trade sensitivity (cost1x, gated)

| window | full | removing single BEST trade | removing single WORST trade |
|---|---|---|---|
| VAL (n=22) | +54.88% | +34.96% (zig075, ret +14.76%, 2025-11-22) | +67.57% (zig075, ret -7.57%, 2025-10-09) |
| OOS (n=24) | +145.34% | +113.62% (zig075, ret +14.85%, 2026-02-26) | +165.84% (zig075, ret -7.71%, 2026-06-25) |

No single trade is the *entire* edge (removing the best trade still leaves a strongly positive
book: VAL +34.96%, OOS +113.62%), but a single trade does move the headline number by 20-35
percentage points in both directions. With only 22-24 trades this is expected and not
disqualifying, but it means the exact reported PnL is not a stable/precise estimate -- treat the
whole VAL/OOS range (roughly VAL 35-68%, OOS 114-166%) as the honest uncertainty band, not the
single point number.

## 3. Rolling walk-forward diagnostic (2025 Q1/Q2/Q3) -- DIAGNOSTIC ONLY, no selection made

| quarter | pnl | mdd | trades | wr | zig075 SHORT sum_ret |
|---|---|---|---|---|---|
| 2025-Q1 | +28.54% | -20.62% | 19 | 0.421 | +0.617 |
| 2025-Q2 | +39.99% | -10.82% | 15 | 0.467 | +0.205 |
| 2025-Q3 | **-9.73%** | **-44.37%** | 19 | 0.316 | **-0.517** |

**This is the most important finding of Phase 1.** Two of three additional quarters are positive
and consistent with the "zig075 SHORT is the edge" story -- but 2025-Q3 (the quarter immediately
preceding the VAL selection window) is net-NEGATIVE overall, with the zig075 SHORT bucket itself
flipping to -0.517, the opposite sign from every other window tested this project (VAL +0.414, OOS
+1.092, Q1 +0.617, Q2 +0.205). MDD in Q3 (-44.37%) is also the worst of any window checked,
exceeding even VAL's -31.11%.

This means the "zig075 SHORT edge" is **regime-dependent, not universal** -- it failed outright in
at least one 3-month window on record. Combined with the small trade counts (Section 2), this
raises the honest possibility that VAL (+54.88%) and OOS (+145.34%) themselves happened to fall in
two favorable regimes back-to-back, and a Q3-2025-like regime could recur at any time live. This is
squarely the kind of risk the roadmap's monitoring/drift-alerting item (Phase 2, item 4) exists to
catch early rather than discover after the fact.

## Bottom line

- Cost robustness: **passes**, first time checked.
- Single-trade dependency: **present but not disqualifying** -- treat reported PnL as a range, not
  a point estimate.
- Regime dependency: **confirmed real** -- 2025-Q3 shows the edge can and does invert for a full
  quarter. This is the single most concrete, quantified problem this project has surfaced this
  session about Omega4.6.1, more concrete than any of the six rejected upgrade candidates. It does
  not by itself argue for reverting the live wiring (VAL/OOS discipline still says keep it -- one
  bad historical quarter among several does not change the pre-registered selection), but it
  substantially lowers confidence that +145.34%/6mo is a reliable forward expectation, and it makes
  Phase 2 monitoring (drift alerting against the realistic range established here) the actual next
  priority, not further architecture search.

Artifacts: `tmp/causal_regen_20260516/omega4_6_1_phase1_robustness_20260707/result.json`. Script:
`scripts/audit_omega4_6_1_phase1_robustness_20260707.py`. Live wiring: unchanged.
