# 5m Feature Audit + Creative Feature Test (2026-07-05)

Diagnostic audit of the user's 125 engineered 5m features + a test of 11 new creative features.
Target = sign of forward 12-bar (1h) return; temporal train (2024+2025H1) / holdout (2025H2)
split; permutation importance (accuracy drop when shuffled). Scripts:
`scripts/analyze_5m_feature_importance_20260705.py`, `scripts/test_creative_5m_features_20260705.py`.
Artifacts in `tmp/causal_regen_20260516/feature_audit_20260705/`.

## Headline: the signal is the problem, not the features

Holdout accuracy for forward-1h direction = **0.5151** with all 125 features (0.50 = coin flip).
Adding the best new features raised it to **0.5187** (+0.36pp). Permutation-importance std is
frequently larger than the importance itself → individual contributions are near noise level.
**There is almost no tree-/linear-extractable 1h-directional edge at 5m** — this is the root
cause of every 5m failure this session (Sigma4, Sigma7), independently confirmed here.

## Existing features: ~46% are dead weight

**58 of 125 features have permutation importance <= 0** (contribute nothing or noise). Category
rollup (sum of positive importance):

| category | sum imp | count(pos) | note |
|---|---|---|---|
| price_action | 0.0138 | 8 | carries the signal |
| other | 0.0118 | 19 | |
| tech_indicator | 0.0093 | 9 | |
| cvd/cvp | 0.0090 | 8 | |
| time | 0.0032 | 5 | |
| volatility | 0.0032 | 6 | redundant cluster |
| btc_cross | 0.0025 | 6 | |
| funding | 0.0017 | 4 | **7 of 11 dead** |
| orderflow | 0.0003 | 2 | **7 of 9 dead — near worthless** |

Top real contributors: cvp_volume_imbalance, hurst_288, fibonacci_level, vwap_dist_288,
turtle_signal, breakout_strength, macd_hist, cvd_288/48, mean_reversion_z, atr_pct_rank_288,
trade_intensity, whale_conviction.

Notable DEAD features (safe to prune): almost all funding (funding_z_score, ou_funding_z,
funding_abs, funding_pressure, mta_funding, funding_flip_signal, funding_roc_288), most orderflow
(smart_money_flow, net_taker_ratio, taker_acceleration, oi_change_rate, oi_up_price_*), event
flags (jump_flag, evt_tail_flag), redundant vol (garch_vol_z, parkinson_vol, garman_klass_vol,
hurst_48), squeeze/liquidity signals (sig_liquidity_trap, squeeze_power, liquidity_vacuum),
and various sweep/failed-breakout flags. 10 redundant pairs (|corr|>=0.9) also exist among the
survivors (e.g. kalman_velocity≈hma_slope≈mtf_trend_1h; volatility_z≈atr_pct_rank_288).

## Creative new features (11 tested): 2 clear wins, most dead

| new feature | rank/136 | verdict |
|---|---|---|
| **dist_lo_atr** (distance to 96-bar low in ATR units) | **3** | strong — top-tier |
| **ret_skew_48** (rolling return skew) | **9** | useful |
| vol_expansion (short/long realized-vol ratio) | 26 | marginal + |
| eff_ratio_12 (Kaufman efficiency, 1h) | 29 | marginal + |
| dist_hi_atr | 37 | ~0 |
| cvd_div / accel_vwap / mtf_slope_agree | 52/55/72 | dead |
| trend_quality / trend_age / eff_ratio_48 | 84/96/104 | dead (neg) |

**Honest surprise**: the hypothesis that a trend-vs-chop discriminator (efficiency ratio, trend
age, MTF agreement) was the missing key was WRONG — those simple versions are dead. The
trend/chop edge that worked in Sigma6 came from a properly-trained HMM regime model, not a ratio.
The genuine find was **dist_lo_atr** (support-proximity in ATR units) at rank 3, plus ret_skew_48.

## Recommendations

1. **Prune the ~58 dead features** (especially funding 7/11 and orderflow 7/9 — 20 features for
   ~0 contribution): reduces noise/overfit surface, speeds training, no signal loss.
2. **Adopt dist_lo_atr and ret_skew_48**; optionally vol_expansion / eff_ratio_12.
3. **But the 51.9% ceiling stays** — feature hygiene cuts noise, it does not create edge. The
   real levers remain: (a) lower decision frequency (1h — already shown to work: Sigma6 OOS
   +45.9%), and (b) new information outside the OHLCV+funding+OI universe (order-book
   microstructure once ~6 months of the live duckdb feed accrue). Feature engineering on the
   existing 5m universe has hit diminishing returns.

## Follow-up: applying the audit to the working 1h Sigma6 model (both directions, both FAILED OOS)

Per user request, tested BOTH (a) adding the audit-winner features and (b) pruning dead-category
features on the actual working 1h pipeline (Sigma3 signal → Sigma6 regime-trend backtest).
`scripts/build_1h_enriched_dataset_20260705.py`, `run_sigma8_enriched_20260705.py`.

| Feature set | lev3 VAL / OOS | lev4 VAL / OOS |
|---|---|---|
| **Original Sigma6 (38 feat)** | +34.3% / **+16.6%** | +71.1% / **+45.9%** |
| (b) Enriched (43 = 38 + 5 audit winners) | +26.9% / -2.3% | +71.5% / +27.7% |
| (a) Pruned+winners (32 = drop 11 funding/OI/weak, keep 2 winners) | **+51.1%** / **-12.1%** | +43.6% / -17.0% |

**Both feature modifications improved-or-held validation but DEGRADED OOS** — the pruned variant
notably jumped validation +34%→+51% while OOS collapsed +16.6%→-12.1%. Textbook overfitting: the
feature changes fit the (already heavily-peeked) validation window better without adding
generalizing signal. Reasons: (1) the audit winners were selected on the 5m raw-direction target
and do not transfer to the 1h trend-scanning/trend-following/regime setup; (2) features "dead" at
5m are not dead at 1h — pruning them lost real information.

**Conclusion**: the original Sigma6 38-feature set is already near-optimal for this task; neither
enrichment nor pruning helped OOS. Consistent with the whole session's lesson — the edge comes
from structure (1h cadence + regime filter + trend-following), not from feature-set fine-tuning.
Recommend keeping the original Sigma6 feature set; do NOT ship the enriched/pruned variants.
