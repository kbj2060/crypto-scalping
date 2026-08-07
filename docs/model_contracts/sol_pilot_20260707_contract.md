# SOL Pilot (Phase A, option A) — infrastructure proven, signal REJECTED

Status: `research_negative_result_not_adopted`. User request 2026-07-07: since ETH trades too
infrequently, build a second coin (SOL) end-to-end "exactly the same way" as Omega4.6.1 to increase
trade count. Scoped to a single-coin pilot first, "properly" (own data/labels/regime/model), with a
quick simplified first-pass validation before committing to a full risk-sidecar build.

## What was built (all successful)

1. **Raw data**: SOLUSDT 5m futures klines (Binance public REST, 2024-06-01..now, 220,680 rows) +
   daily OI/top-trader metrics + monthly funding rate (data.binance.vision public archive, same
   source already used for ETH) -- confirmed available for SOL, not an ETH-only limitation.
   Scripts: `download_klines_sol_20260707.py`, `download_metrics_funding_sol_20260707.py`,
   `build_sol_raw_frame_20260707.py`.
2. **Feature engineering**: the IDENTICAL `features/engineering.py::FeatureEngineer` pipeline used
   for ETH ran clean on SOL with zero errors and zero all-NaN columns (142 columns, 220,679 rows,
   2024-06-01..2026-07-07). Script: `build_sol_features_20260707.py`.
3. **Regime3 HMM**: a SOL-specific regime3 (current_sensitive/wide24, bull/bear/chop) HMM trained
   with the identical builder (`experiment_regime3_current_hmm_wide24_20260529.py`), balanced
   accuracy 0.71 -- healthy, not degenerate. Directly refutes the earlier assumption (carried over
   from Sigma9, a different lineage) that a working regime filter can't be built for a non-ETH
   asset; it was simply never attempted for this architecture before.
4. **Labels**: zigzag_action labels built with the identical barrier-touch algorithm and params.
   SOL's raw label balance (2025: LONG 18,367 / SHORT 18,499, ratio 1.01) closely matches ETH's
   (0.99) -- confirms the earlier finding that neither asset's raw label distribution explains the
   live model's SHORT bias (see chat record; not a data-availability question).

**Conclusion: "완전히 동일한 방식" (exactly the same way) is genuinely achievable for a new asset.**
This is the main positive, reusable result of this pilot -- the scripts above are a template for any
future asset addition.

## Quick first-pass signal validation — REJECTED

Per explicit user scope choice ("빠른 1차 검증"), skipped the full risk-sidecar/exit-head build and
used: static TP7.5%/SL4% barrier (zig075's own values), fixed margin_fraction=0.30/leverage=3x, no
exit-head, no duration gate, single component. TRAIN 2025-01-01..09-30, VAL 2025-10-01..12-31
(threshold selection), OOS 2026-01-01..06-30 (one-shot). Two variants tested:

| variant | base_cols | VAL (best threshold) | OOS (frozen, one-shot) |
|---|---|---|---|
| v1 | all 141 engineered columns | +47.87% (th=0.60) / MDD-20.17% / n=39 | **-9.55% / MDD-49.82%** / n=61 |
| v2 | zig075's exact 102 base_cols | **all 4 thresholds negative** (best: -10.92% at th=0.75) | +27.62% / MDD-15.10% / n=18 (NOT actionable -- VAL never cleared) |

v1: classic VAL-good/OOS-collapse pattern, same signature as Candidates 1/5/6/7 on the ETH side
this session. v2: VAL itself never produces a profitable configuration, so under this project's
pre-registered discipline (VAL selects, OOS confirms once, no peeking after a VAL failure) the
one-shot OOS number is **not usable as promotion evidence** even though it happened to be positive
-- exactly the kind of result the discipline exists to prevent people from rationalizing into a
false promotion.

**Verdict: REJECTED.** Neither variant of this quick, simplified SOL model shows a disciplined,
VAL-passing edge. This does not prove SOL has no exploitable edge (the barrier/sizing/exit-head
were deliberately simplified relative to Omega4.6.1's full stack, and trade counts here are as thin
as ETH's own early history), but there is no basis to invest further (full risk sidecar, proper
duration gate calibration) at this time.

## If revisited later

- Try a different coin (BNB/XRP were the other candidates discussed) using the exact same
  infrastructure template built here -- steps 1-4 above are directly reusable with a new symbol.
- Or build the FULL stack for SOL (risk sidecar, proper exit-head, duration gate) despite this
  quick-check result, but only with a clear reason to expect the full stack would behave
  differently from the simplified one (not established here).
- Either path should still respect the trade-count scarcity lesson learned throughout this
  session: a handful of dozens of trades is not enough to reliably validate a new model, asset, or
  architecture change.

Scripts: `download_klines_sol_20260707.py`, `download_metrics_funding_sol_20260707.py`,
`build_sol_raw_frame_20260707.py`, `build_sol_features_20260707.py`,
`split_sol_features_by_year_20260707.py`,
`experiment_regime3_current_hmm_wide24_20260529.py` (reused, SOL args),
`build_sol_zigzag_labels_20260707.py`, `train_eval_sol_quick_pilot_20260707.py`. Live wiring: NONE
(trading_bot.py untouched throughout; Omega4.6.1 remains the sole live model).
