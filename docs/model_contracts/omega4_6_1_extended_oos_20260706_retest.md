# Omega4.6.1 duration_ou_halflife_risk_gate — Extended OOS Retest (2026-01-01..06-30)

Status: `research_retest_positive_but_feature_drift_caveat`

Last updated: 2026-07-06 KST

User request: `omega4_6_1_duration_ou_halflife_risk_gate_20260630` (see
`docs/model_contracts/omega4_6_1_duration_ou_halflife_risk_gate_20260630_contract.md`) was
identified as "good returns but low trade count" and the user asked to retest now that OOS data
extends past the original 2026-01-01..02-28 window through 06-30.

## Methodology: frozen artifacts re-scored on extended data, NO retraining

The model is a deep chain: two independently-trained 3-head TabM parents (`h48qual` q=0.50,
`zig075` q=0.75) → two frozen risk sidecars (`risk_sidecar.pkl`, HGB margin/leverage regressors)
→ priority router (h48qual > zig075) + fixed rescale map → frozen duration gate
(`ou_halflife <= 0.005415348` → skip). All of these are FROZEN, untouched artifacts; only the
input data changes.

Investigation found the parent models need **zero m7/NF features** (only 102 OHLCV/technical/
regime3 columns), so the m7-unrecoverable problem that blocked the original Omega6 v1 winner does
**not** block this model. However, the original scoring used a legacy feature file
(`trade_candidates_2026_alpha6_current_tail111_exact.csv`, Jan-Feb only) that differs from the
current canonical `training_features_2026_rebuilt.csv` on 5 of 96 columns (correlation on the
Jan-Feb overlap): `ou_halflife` (corr **-0.03**, essentially unrelated), `kel` (0.62),
`evt_excess_z` (0.79), `btc_corr_60` (0.85), `dual_momentum` (0.93). The other ~91 columns
correlate >=0.99. Root cause is most likely a `features/elite.py` formula change since
2026-05-29; git history for that file is too sparse (3 commits total) to recover the exact old
version. Per user direction, the ENTIRE Jan-Jun window (not just the new March-June tail) was
recomputed uniformly from the current pipeline to avoid splicing two inconsistent feature
vintages at the Feb/Mar boundary — this means the Jan-Feb portion here is not bit-identical to
the originally published numbers either, but the whole window is internally consistent.

**`ou_halflife` is exactly the feature the duration gate depends on** — this is the single most
important caveat on this retest's result.

Scripts: `scripts/build_omega4_6_1_extended_parent_predictions_20260706.py` (parent inference),
`scripts/retest_omega4_6_1_extended_oos_20260706.py` (risk-sidecar sizing + exit-head replay,
reusing `_replay_with_risk`/`_risk_margins`/`_risk_leverage` imported directly from the original
training scripts), `scripts/combine_omega4_6_1_extended_oos_20260706.py` (router + duration gate).

## Self-check: reimplementation validated against published numbers

Before trusting the extended result, the same code was run on the ORIGINAL Jan-Feb alpha6/7
frame and compared to the already-published component numbers:

| Component | Metric | Published | Self-check (this reimplementation) |
|---|---|---|---|
| h48qual | OOS pnl / mdd / trades / wr | 13.39% / -4.43% / 8 / 0.625 | 13.56% / -4.68% / 8 / 0.625 |

Trade count and WR match exactly; PnL/MDD are within ~0.3pp (residual due to the cost_mult=1.0
assumption and the same feature-drift columns already present even in the original window). This
validates the reimplementation before applying it to new data.

## Result: PnL up, trade count roughly doubled, MDD modestly worse

| Window | PnL | MDD | Trades | WR |
|---|---|---|---|---|
| **Original (Jan-Feb only, published)** | +72.59% | -7.47% | 13 | 66.7% |
| **Extended (Jan-Jun, this retest, post duration-gate)** | **+145.46%** | **-10.82%** | **25** | **52.0%** |
| Extended, pre-duration-gate (router combine only) | +141.14% | -13.78% | 33 | 48.5% |

Monthly breakdown (post-gate): Jan +25.5% (7 trades, wr 0.57), Feb +29.5% (2, wr 1.00), Mar +26.3%
(6, wr 0.50), Apr +2.7% (2, wr 0.50), May +21.7% (2, wr 1.00), Jun **-5.4%** (6, wr 0.17). Five of
six months positive; June is the only clear drawdown month, driven by a lower win rate rather than
one catastrophic trade.

The duration gate skipped 8 of 33 candidate trades (`hit_count=8`) in the extended window, raising
WR from 48.5%→52.0% and PnL from +141.1%→+145.5% while cutting MDD from -13.8%→-10.8% — consistent
with its original design intent (veto short-duration-reversion setups). Because `ou_halflife`
itself has near-zero correlation with the original feature vintage, this specific improvement
should be read with more caution than the underlying router-combine result.

## Verdict

Directionally positive: extending the OOS window nearly doubles the trade count (13→25) while
PnL improves and MDD stays within the project's ~20% absolute bound — the "too few trades to
trust" concern is meaningfully, if not completely, alleviated (25 trades over 6 months is still a
modest sample for a leveraged strategy). The result is **not a clean, bit-identical continuation**
of the original scoring due to the documented `ou_halflife`/`kel`/`evt_excess_z`/`btc_corr_60`/
`dual_momentum` feature-vintage drift — most directly relevant to the duration gate's specific
contribution, less so to the underlying router-combine PnL (which only depends on the ~91
well-correlated columns via the parent models). Recommend: (1) treat this as a strong but
not-fully-clean confirmation, (2) if promoting, first pin down why `features/elite.py`'s
`ou_halflife`/`kel`/`evt_excess_z` formulas diverged from the alpha6/7-era snapshot, since that
gap will recur for any future extension of this lineage.
