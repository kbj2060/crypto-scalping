# Sigma6 — Regime-Filtered Trend-Follower (BEST generalizing result)

Status: `research_strongest_oos_result_pending_fresh_window_confirmation`

Last updated: 2026-07-05 KST

Lineage: Sigma5 "let winners run" trend-follower (on the Sigma3-1h ensemble signal) + a REGIME
FILTER reusing the Omega stack's Regime3 classifier — per user request to "reference good
techniques from the Omega models and add a regime filter."

## Design

- **Signal**: Sigma3-1h ensemble (5-seed HGB on 1h trend-scanning labels), unchanged.
- **Execution**: Sigma5 trailing-stop trend-follower (trail 5xATR give-back, hard stop 2.5xATR,
  min-profit-arm 2xATR, max_hold 144 bars = 6 days, compounding, leverage 3-4x).
- **Regime filter (the Omega technique)**: the Regime3 'current' HMM nowcast
  (`regime3_current_sensitive_hmm_wide24`, bull/bear/chop probs, trained on 2024, applied
  causally) merged onto the 1h tape (`merge_asof` backward, causal), plus the CryptoMamba-h6
  `stability_score` (skip transitioning regimes). Winning mode = **`not_chop`**: enter a trade
  only when `chop_prob < 0.42` AND `stability_score >= 0.55` — i.e. only trend-follow when the
  market is in a stable, non-choppy (= trending) regime, and sit out chop entirely.
- Script: `scripts/run_sigma6_regime_trend_20260705.py`. cost1 primary.

## Result: the regime filter turned Sigma5's OOS failure into a strong OOS pass

| Config | VAL cost1 | **OOS cost1** | OOS MDD | OOS WR |
|---|---|---|---|---|
| Sigma5 no filter (lev3) | +118.3% | **-2.5%** | -25.0% | 30.8% |
| **Sigma6 not_chop+stab (lev4/sl2.5)** | +71.1% | **+45.9%** | -15.1% | 44.4% |
| **Sigma6 not_chop+stab (lev3/sl2.5)** | +34.3% | **+16.6%** | -16.0% | 50.0% |
| Sigma6 trend_agree (lev4) | +78.9% | -24.8% | -36.4% | 26.9% |

OOS = 2026-03-02..06-30. The `not_chop`+stability filter cut the choppy-period bleed exactly as
intended: OOS monthly (lev4) Mar +28.8, Apr -8.3, May -0.8, Jun +24.6 — the down months are
shallow (the filter keeps the strategy out of the conditions where a low-WR trend-follower loses)
while the trending months (Mar, Jun) delivered big. lev3 is steadier (WR 50%, OOS +16.6%).

Note the `trend_agree` mode (require bull-regime for longs / bear for shorts) FAILED (-24.8%) —
too restrictive and mis-timed; the value is specifically in **avoiding chop**, not in matching
the regime's directional label.

## Verdict: strongest generalizing result in the project — but confirm on a truly fresh window

This is the first approach where a high-leverage trend-follower survived the fresh window with
strong positive returns (OOS +45.9% at lev4 / +16.6% at lev3, both MDD ~-15%). The mechanism is
principled (trend-follow only in stable non-chop regimes) and holds across leverage settings,
which argues against it being a single lucky point.

**Honest caveats (do not overclaim):**
1. The 2026-03-02..06-30 window has now been scored multiple times (Sigma3, Sigma4, Sigma5, and
   4 Sigma6 configs). Its evidential value as "unseen" is degraded. The regime-filter parameters
   were selected on VALIDATION (not OOS), and the mechanism is principled, but the honest final
   test requires a genuinely fresh window — **2026-07+ as it accumulates**.
2. Returns are still trend-following in nature (2 of 4 OOS months negative, profit from the
   trending months); expect flat-to-down stretches in extended range-bound regimes even with the
   filter.
3. Leverage 3-4x with compounding drives the headline numbers; at 1x notional the edge is smaller.

**Recommended live candidates** (if evaluated at realistic cost1 fees):
- **Conservative**: Sigma6 lev3 not_chop+stab — OOS +16.6%, WR 50%, MDD -16%.
- **Aggressive**: Sigma6 lev4 not_chop+stab — OOS +45.9%, WR 44%, MDD -15%.

**Next steps**: (1) hold out 2026-07+ untouched and score this frozen config once when ~2-3
months accrue — that is the real confirmation; (2) add the order-book microstructure regime
signals (duckdb, once ~6mo history) to sharpen the chop/trend classification further; (3) consider
a small ensemble over regime thresholds to reduce parameter sensitivity before live wiring.

## Contamination / lookahead audit of the regime filter (2026-07-05)

Full code-path audit + empirical lag test. **Conclusion: no lookahead or distributional leak;
but the result is timing-fragile, so the honest live estimate is below the headline.**

Code-path (all clean):
- **Regime3 wide24 HMM**: trained on 2024 only (`DEFAULT_TRAIN_2024`). `_transform` on 2025/2026
  uses the payload-FROZEN scaler/medians/model/state_class_matrix — NOT refit on test data (no
  distributional leak). Inference is `GaussianStateModel.filter_proba` = forward-only alpha
  recursion (`log_alpha[t]` from `log_alpha[t-1]`+trans+emit[t]); verified NOT a forward-backward
  smoother, so proba[t] depends only on obs[0..t]. Causal.
- **CryptoMamba-h6 stability**: trained on 2024. `SeqDataset` window is strictly trailing
  (`x[end-seq_len+1 : end+1]`). The forward-horizon regime is the TRAINING TARGET only (inference
  passes y=None). `stability_score` = model's PREDICTED persistence probability (a forecast from
  past features), not a realized future outcome. Causal.
- **Merge**: `merge_asof(..., direction="backward")` attaches regime with ts <= tape ts; the 1h
  decision executes at the next-bar open (H+1:00) while the merged regime is from H:00 — i.e. the
  regime is if anything STALER than the decision, never ahead of it. Causal.
- **Extended 2026 regime files**: built by applying the frozen 2024 models, reproducibility-
  verified earlier (wide24 5.6e-16, cmamba 2.4e-07 on the Jan-Feb overlap). OOS regime columns
  have 0 NaN and the filter removes 71.5% of OOS bars (chop_prob median 0.49, stability median
  0.63 — meaningful variation, not a trivial always-on/off filter).

Empirical lag test (OOS 2026-03..06, lev4 config), shifting the regime signal to STRICTLY OLDER
bars: same-bar **+45.9%** → +1 bar (1h) **+29.8%** → +2 bar **+23.8%** → +3 bar (3h) **-22.2%**.

Interpretation: the same-bar regime is causal (known before the next-bar-open execution), so the
degradation-with-lag is the legitimate/causal fact that fresher regime is more informative — NOT
lookahead. BUT the sign-flip by 3h staleness shows the edge is **fragile / timing-sensitive**.
The headline +45.9% assumes freshest-available regime; a latency-robust live estimate is the
+1-2 bar range (**+24% to +30%**). Since the HMM nowcast computes in seconds on completed 5m
bars (sub-1-bar real latency), ~+30% is the reasonable expected figure — still strongly positive,
but the +45.9% should be treated as an optimistic ceiling, not the point estimate. This
fragility is an additional reason to (a) confirm on the fresh 2026-07+ window and (b) build the
regime-threshold ensemble before any live sizing.
