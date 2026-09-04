# ETH regime GBM2 (trend/chop) — 2026-08-27

## Outcome

Deployed. Replaces GBM3 (bull/bear/chop, 2026-08-26) as the Snapshot tab's
liquidation-map chart regime ribbon. Dashboard-only — does not touch
`trading_bot.py`'s live `RegimeEngine`-based owner routing.

Motivation: the user found the 3-class ribbon flickered too much to trust for
a discretionary chop->trend repositioning decision. A 3rd discrete
"transition" class was considered and rejected — see Policy conflict below.
Instead the classes were collapsed to 2 (trend = bull+bear merged, chop) and
the training *label itself* (not just serving-side smoothing) was stabilized
with a debounce filter, after the user reviewed the raw label plotted
against real OOS price and asked for it to flip less.

## Policy conflict considered and avoided

A 3-class (trend/transition/chop) redesign was the first idea discussed, but
this repo already tried and rejected treating instability as a discrete
class twice:
- `docs/model_contracts/regime3_whipsaw_risk_policy_20260529.md`: forbids a
  discrete whipsaw/transition class for new action classifiers (OOS showed
  frequent flips that destabilize the classifier and conflict with chop).
- `docs/active_live/regime3_policy_20260530.md` (still the active policy):
  a continuous transition-risk score reached only 2026 OOS AUC=0.676 /
  bal_acc=0.587 — "not reliable enough to own future class direction."
- The 2026-08-26 GBM3 whipsaw-hierarchical research independently
  re-derived the same conclusion after 6 rounds (see memory
  `eth_regime_hierarchical_whipsaw_circularity_rejected_20260826`).

Merging bull+bear into one "trend" class was never tried in any of those
three rounds and does not introduce a new discrete instability class, so it
does not reopen this policy conflict.

## Label contract

- Base rule: `features.elite.RegimeEngine.compute()` (unchanged, the same
  rule engine `trading_bot.py`'s live owner routing already consumes).
  `is_trend_raw = regime_bull | regime_bear`; `chop = regime_chop |
  regime_whipsaw | regime_normal` — confirmed to match GBM3's own merge by
  comparing GBM3's in-sample predicted class distribution (bull 25.86% /
  bear 24.22% / chop 49.93%) against this raw split on the same TRAIN rows
  (24.29% / 22.71% / 53.00%).
- Stabilization: discrete K-consecutive-bar debounce (`_debounce()` in
  `scripts/train_eth_regime_gbm2_trend_chop_20260827.py`) applied to
  `is_trend_raw` — **the debounced sequence, not the raw label, is the
  actual training target.** `k_bars=12` (1h at 5m bars) chosen after the
  user reviewed `is_trend_raw` plotted against OOS price (flip_rate=0.1877,
  visibly flickery inside chop ranges) and a k in {6,12,24,48} comparison
  chart: k=48 (4h) visually locked into one state for a 5-day test window
  (trend_share collapsed 0.45->0.12, a debounce-mechanism failure, not
  genuine stability); k=24 (2h, flip_rate=0.0023) was picked from the
  numeric grid alone first, then the user switched to k=12 (flip_rate=0.0128
  full-OOS) after seeing the side-by-side chart and preferring its finer
  responsiveness with no lock-up symptoms.
- Full-range (2024-01-01~2026-08-19) flip_rate: raw 0.1991 -> confirmed
  (k=12) 0.0123.

## Training contract

- Features: GBM3's 136 `feature_cols` reused verbatim (already excludes the
  5 columns confirmed circular with the RegimeEngine label formula:
  `mtf_trend_1h`, `state7_trend_efficiency_48`,
  `state7_directional_return_48`, `state7_volatility_state`,
  `state7_sign_flip_rate_24`).
- Data: `data/splits/year_oos/training_features_{2024,2025,2026_rebuilt}.csv`
  concatenated (the only viable source — live Binance backfill is
  physically capped at ~30 days by `startTime` validation, confirmed by
  direct API test, so it cannot reach 2024).
- Split: TRAIN 2024-01-01~2026-06-30 (final fit, matches GBM3), internal
  causal VAL 2026-04~06 (HP/hysteresis selection only, never leaked into the
  selection-phase fit), OOS 2026-07-01~08-19 (confirmatory). **This OOS
  window has been reused across ~8 prior regime-classifier rounds in this
  repo (wide24 grid sweeps, whipsaw's 6 rounds, GBM3 itself) — reported
  honestly here, not claimed single-touch-pure.**
- Model: `HistGradientBoostingClassifier(max_depth=10, learning_rate=0.04,
  max_iter=400, l2_regularization=2.0)` — GBM3's own config, confirmed still
  competitive via a 3-way VAL-only check (0.7027 vs 0.7033 vs 0.6974 bal_acc
  against the debounced target on TRAIN-minus-VAL fits) before keeping it.
- Class encoding: int-coded `y` (`model.classes_ == [0, 1]`), asserted at
  train time — GBM3's own joblib was found to rely on the same convention
  (its `classes_` are ints, `payload["classes"]` a separate parallel string
  list); fitting on raw strings would let sklearn silently alphabetize
  classes and scramble `trend_prob`/`chop_prob` in production.

## Results

| split | metric | argmax (raw) | + serving hysteresis (k=3, band=0) |
|---|---|---|---|
| VAL (partly in-sample) | balanced_accuracy | 0.8823 | 0.8907 |
| VAL | flip_rate | 0.0489 | 0.0185 |
| OOS (held out) | balanced_accuracy | 0.7669 | 0.7812 |
| OOS | flip_rate | 0.0604 | 0.0203 |

The serving-side hysteresis (`_apply_hysteresis()`, VAL-selected 2-parameter
grid over k_bars in {1,3,6} x band in {0.0,0.05}) is a secondary pass on top
of the model's own probability output — it exists because the model does not
fully reproduce the debounced label's own low flip-rate (0.0128 on OOS)
purely from learning the target; some residual boundary noise remains in
`predict_proba` and this filters it further.

**OOS bal_acc (0.78) is well below GBM3's (0.9189) — this is a genuinely
harder target, not a modeling regression.** A causal check (predictions
bucketed by distance-in-bars to the nearest confirmed-label transition)
shows error concentrates almost entirely near transitions: 44% error rate
within 3 bars of a flip, falling to 15% error beyond 48 bars (4h) into a
stable regime. 28.5% of all OOS bars sit within the 12-bar debounce window
of some transition, which is intrinsically the hardest zone for any
same-bar-features model to call, since even the label itself hasn't
"confirmed" yet at that point. Far from a transition — the situation the
user actually wants to trust before repositioning — the model is ~85%
accurate.

## Post-deploy override (same day)

The user reviewed a side-by-side of GBM3's actual OOS predictions vs GBM2 raw vs GBM2 confirmed
(same price window) and decided immediate reaction matters more than the extra smoothing pass, once
it was concrete that GBM2's raw output (no serving-side hysteresis) is already far more stable than
GBM3 was (flip_rate 0.060 vs 0.182/0.193 on this same OOS window) purely because the *model* was
trained on the k_bars=12-debounced label — the serving hysteresis was only ever a secondary layer on
top of that. `payload["hysteresis_config"]` was overridden from the VAL-selected `k_bars=3` to
`k_bars=1, band=0.0` (disables the secondary smoothing pass entirely; `raw_state` and
`confirmed_state` are now always equal). No retrain, no code change — the serving script re-reads
this config from the joblib on every call, so pushing the updated artifact took effect immediately.
Reverting is a one-line config edit + `handoff.sh push`, not a retrain, if faster reaction turns out
too noisy in practice.

## Deployment

- Artifact: `tmp/eth_regime_gbm2_trend_chop_20260827/model.joblib` (full
  provenance in the payload: feature list, label logic, both merge
  verification and hysteresis grids, OOS-reuse caveat — GBM3's own training
  script was lost after deployment; this one is committed as a permanent
  repo file specifically to not repeat that).
- Serving: `scripts/live_regime_gbm2_trend_chop_signal_20260827.py`, a
  structural copy of `live_regime_gbm3_signal_20260826.py` (same fetch
  helpers, same `FeatureEngineer`+`_with_raw_state12` pipeline). Fully
  stateless — recomputes raw+confirmed states from scratch over the whole
  15-day fetch window every call, no persisted state needed.
- Output contract: `{warmed_up, error, latest_bar_utc, trend_prob,
  chop_prob, confidence, raw_state, confirmed_state, bars_since_confirm,
  history}` — drops `bull_prob`/`bear_prob` rather than faking them.
- `dashboard/server.py`: single import swap (`compute_regime_gbm2_trend_chop_signal`
  aliased to `compute_regime_wide24_signal`, matching GBM3's own cutover
  pattern — routes/cache/variable names unchanged).
- `dashboard/live/app.js`: `REGIME_DOMINANT_COLOR` (bull/bear/chop ->
  trend/chop, new trend color `#3b6fd6` chosen to not collide with the old
  bull-green/bear-red), `regimeDominant()` now prefers the server's
  `confirmed_state`, both regime tooltips updated to show a "(확인중)" tag
  when a bar's `raw_state` differs from its `confirmed_state`.
- Verified live: `/api/regime-wide24` on the server returns the new
  contract shape (curl-checked), served `index.html`/`app.js` carry the new
  cache-buster and edited code. No browser available on the dev machine —
  verified via served bytes + live API data trace, not visual rendering.
