# BTC 3-way TP/SL/timeout ("TP-first") Label — CLOSED 2026-08-04

## What was tried

User proposed a 2-branch fusion network (94 market + 16 context [Regime3 CURRENT 4 +
DVOL 6 + on-chain 6] -> concat -> residual fusion -> Long/Short TP-first heads) and
asked to research the label first, since "라벨이 제일 중요해." Before building the
architecture, per the project's cheapest-falsification convention, the label alone
was tested through the existing dense-nogate LightGBM pipeline.

**Prior art found first:** `docs/model_reports/btc_v2_direction_meta_20260716.md`
already ran an extremely broad meta-labeling search (12,544 threshold/model/execution
candidates across fixed-horizon, dollar-event, directional-change, denoised-SSL,
prior-meta-label, reward-shaping, zigzag, and trend-scan label families, including a
"terminal target" and "execution-aligned target" essentially equivalent to TP-first
classification) on BTC v2's own F0/F1 microstructure features. **0/12,544 passed the
40-trades + non-negative-3x-cost gate.** User chose to still cheaply re-test the idea
on the *current* feature set (causalfix_final + Regime3 wide24 + DVOL + on-chain,
unified in `data/splits/year_oos/btc_unified_raw_panel_20260804.parquet`) rather than
skip it outright, and to handle the timeout case as an explicit third class (3-way
TP/SL/timeout) rather than dropping timed-out samples (avoids the survivorship-bias
failure mode of BTC v2's "terminal target" family).

## What was run

`scripts/eval_btc_3way_tpfirst_label_cheap_falsification_20260804.py`: per side
(long/short), 3-class LightGBM classifier (`sl`/`tp`/`timeout`, via the same
`_reason_and_return` barrier-touch function used throughout the project's triple-
barrier labels, unmodified). Trading score = `P(tp) - P(sl)`. Same VAL/OOS split
(2025-09-01 / 2026-01-01 to 2026-04-01), same conservative cost model, same
`n_trades>=15` pass bar as every prior stage in this line.

## Result: CLOSED, 0/24 configs pass

Both calibrations (h48qual_shape 48-bar, longhold_shape 576-bar) x 6 thresholds each,
VAL and OOS: **no config is VAL+OOS both positive.** More notably, `mean_net_pct`
stays essentially flat (~-0.40% to -0.50%) across every threshold from 0.0 to 0.3 in
both splits -- unlike the regression-label baselines (Stage A/B in
`docs/btc_deepfeat_jepa_unified_panel_closed_20260804.md`), which at least showed some
threshold-dependent movement, the `P(tp)-P(sl)` score here shows **no visible
monotonic relationship with realized trade quality at all**, i.e. the classifier is
not extracting a usable ranking signal, not just missing a profitable cutoff.

## Verdict

This reconfirms BTC v2's 0/12,544 meta-labeling result on an entirely different,
richer feature set (causalfix_final+Regime3+DVOL+on-chain vs BTC v2's F0/F1), with an
explicit 3-way timeout class this time. The proposed 2-branch fusion architecture was
**not built** -- the label it would have been trained on doesn't carry a usable
signal on this feature set, so building the network would not have been informative.

Per the project's overall 2026-08-04 diagnostic (see
`docs/btc_new_architecture_session_summary_20260804.md`,
`project-btc-20260804-session-arc-summary` memory): every architecture, every model
family, every recent data source, a self-supervised deep-feature encoder, and now
both magnitude-regression and TP-first-classification label paradigms have all hit
the same wall on this instrument/timeframe. The remaining open question is whether
the barrier/horizon *calibration* itself (TP/SL multiples, horizon length) rather
than the label *paradigm* is the lever left untried -- or whether this is a genuine
efficiency ceiling for BTC 5m at this feature richness.

## Artifacts

- `scripts/eval_btc_3way_tpfirst_label_cheap_falsification_20260804.py`
- `tmp/btc_3way_tpfirst_cheap_falsification_20260804.csv`
