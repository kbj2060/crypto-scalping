# BTC adaptive_squeeze fresh-forward test -- negative result (2026-07-20)

## Motivation

[`sol_adaptive_squeeze_v2_20260720.md`](sol_adaptive_squeeze_v2_20260720.md) fixed a real
ETH-calibrated funding-rate divisor bug for SOL and reasoned, on distributional grounds alone,
that BTC didn't need the same fix (BTC's `last_funding_rate` std, 0.000041, is already close to
ETH's 0.000044). This test replaces that statistical argument with an actual empirical
fresh-forward retrain, in case the distributional similarity argument was missing something the
SOL case didn't need to check (SOL's std was 3.5x off, an obvious flag; BTC's is not, but "close"
isn't the same as "verified").

## Method

Reproduced the exact same 4-step pipeline used for SOL v2, substituted for BTC, changing only
`FeatureEngineer(adaptive_squeeze=True)` -- same architecture, labels, hyperparameters, quality
mode (`quality_label_action` against `btc_h48_conservative_padded_to_zigzag_timestamps_20260708`),
quality threshold (0.55), exit threshold (0.95), risk-sidecar contract (`parent_outputs`,
side-split, dynamic leverage, `log_risk`/`validation_only` selection), and BTC v1's own
already-selected scale-map (`long_scale=0.5, short_scale=2.5`, `LEVERAGE_CAP=5.0`,
`NOTIONAL_CAP=1.8`) -- reused rather than re-gridded, since this is a same-asset feature-variant
comparison, not a cross-asset transfer.

1. `scripts/build_btc_features_adaptive_squeeze_20260720.py` -> `data/splits/year_oos_adaptive_squeeze_btc_20260720/btc_features_{2025,2026}.csv`
2. `scripts/train_eval_omega4_3head_parent72_loose_entry_quality_btc_adaptive_squeeze_20260720.py` -> `tmp/causal_regen_20260516/btc_omega4_3head_parent72_loose_entry_quality_20260708_h48qual_adaptive_squeeze_20260720/`
3. `scripts/train_eval_omega4_2_risk_sidecar_btc_adaptive_squeeze_20260720.py` -> `tmp/causal_regen_20260516/btc_omega4_2_trade_risk_sidecar_20260708_h48qual_q055_adaptive_squeeze_20260720/`
4. `scripts/apply_final_scale_map_btc_adaptive_squeeze_20260720.py` -> `tmp/causal_regen_20260516/btc_final_scale_map_adaptive_squeeze_20260720/report.json`

All four steps are genuine bar-by-bar causal replays (`fresh_forward_bar_by_bar=true`,
`trade_ledgers_used_as_input=false`, `saved_parent_exit_timestamps_used=false`,
`future_rows_used_for_entry=false`), same val/OOS split convention as BTC v1
(`SPLIT_TS=2025-10-01` train/val boundary within 2025, `oos_frozen_q1_2026` = entries before
2026-04-01) so the comparison to v1 below is apples-to-apples on the same evaluation
methodology BTC v1 itself was scored with.

## Result: clearly worse, not neutral

| | v1 (live, pre-change) | adaptive_squeeze |
|---|---|---|
| VAL, no duration-gate | +7.45% / -11.93% (16 trades) | **-12.28% / -24.68%** (17 trades) |
| VAL, with duration-gate (selected on VAL only) | +12.39% / -6.49% (10 trades) | **-2.77% / -21.94%** (13 trades) |
| OOS-extended, no gate | +22.69% / -15.88% (30 trades) | +30.08% / -19.10% (37 trades) |
| OOS-extended, with gate | +29.23% / -10.65% (24 trades) | **-13.22% / -24.95%** (29 trades) |
| OOS-frozen-Q1-2026, with gate | +10.17% / -10.65% (16 trades) | **-17.72% / -21.87%** (21 trades) |

VAL performance collapses under the fix (both gated and ungated), which matters beyond VAL itself:
the duration-gate threshold is *selected on VAL only* (this project's own no-peeking discipline),
so a worse VAL ranking drags the selected threshold to a worse operating point, which then also
tanks OOS even though OOS's *ungated* number alone looked slightly better in raw PnL (offset by
worse MDD). Every gated number -- the ones that actually matter, since v1 is deployed gated --
moved from solidly positive to solidly negative.

## Conclusion

This empirically confirms and strengthens the prior distributional argument in
`sol_adaptive_squeeze_v2_20260720.md` line 38-40: **adaptive_squeeze must NOT be applied to BTC.**
It is not merely "unnecessary" (neutral) as the earlier statistical read suggested -- it is
actively harmful to BTC's current live model when tested through a real retrain. No live code
changes were made or needed (`trading_bot.py`'s `FeatureEngineer(adaptive_squeeze=(_asset_key ==
"sol"))` already excludes BTC; this test only produced new scratch artifacts under
`data/splits/year_oos_adaptive_squeeze_btc_20260720/` and `tmp/causal_regen_20260516/btc_*adaptive_squeeze_20260720/`,
none of which are referenced by any live config).

This closes the "should we transfer SOL's fix to BTC" question definitively -- do not re-run this
test again absent a materially different feature/architecture change.
