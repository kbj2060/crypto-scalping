# Funding Feature Red-Team Audit - 2026-05-29

## Verdict

Active/live funding cleanliness is not safe by default. The current trading bot still points to the stale Alpha7 decontam artifact family, while the safe clean-funding path is research-only and not live-wired.

The pre-clean Alpha7 `01965` artifact families are deprecated and blocked for active/candidate reuse:

- `data/ensemble/supervised/alpha7_1_01965_live_20260527`
- `data/ensemble/supervised/alpha7_submodel_01965_decontam_v2_tp_20260528`

Both directories now contain `DEPRECATED_DO_NOT_USE.json`; their `alpha7_live_manifest.json` status is `deprecated_do_not_use_active_or_candidate`.

The clean `01965` rebuild also needed one additional fix: columns whose names do not contain `funding` but are derived from funding, such as `squeeze_power`, `long_squeeze_risk`, and `crowding_pressure`, must be overlaid from clean feature frames instead of inherited from the old candidate skeleton or overwritten by generic M7 frames.

## Funding Feature Family

Audit scope includes:

- `last_funding_rate`
- `funding_abs`
- `funding_pressure`
- `funding_roc_12`
- `funding_roc_48`
- `funding_roc_288`
- `funding_z_score`
- `funding_price_divergence`
- `long_squeeze_risk`
- `short_squeeze_risk`
- `squeeze_power`
- `mta_funding`
- `ou_funding_z`
- `crowding_pressure`
- `sig_ai_squeeze`
- any model artifact trained or scored from those inputs

`funding_rate_sign` was searched but was not present as an active feature symbol.

## Blocking Findings

1. `trading_bot.py` defaults still reference `data/ensemble/supervised/alpha7_submodel_01965_decontam_v2_tp_20260528`. That artifact manifest does not document clean funding remediation and points to stale `alpha7_1_01965_v2only_tp_sl_action_score_20260528`.
2. Pre-clean Alpha8 result files using `baseline_model_id=alpha7_submodel_01965_decontam_v2_tp_20260528` remain invalid for promotion, even if their direct CSV inputs were later partially cleaned.
3. `alpha8_final_candidates_verified_20260529.json` has regime checks but no funding-clean provenance, so it is not promotion evidence.
4. The stale Alpha7 candidate CSVs under `tmp/causal_regen_20260516/alpha7_1_01965_v2only_tp_sl_action_score_20260528/` materially differ from the clean-funding candidate frames and remain research-only.

## Quantified Stale Difference

Old 2025 candidate versus stricter clean candidate:

- `last_funding_rate`: max diff `0.00140792`, diff rows `105064`
- `funding_pressure`: max diff `0.20464266`, diff rows `105064`
- `funding_roc_288`: max diff `7.28337166`, diff rows `96814`
- `funding_price_divergence`: max diff `6.0`, diff rows `20222`
- `squeeze_power`: max diff `8348629.44`, diff rows `105055`
- `mta_funding`: max diff `2.0`, diff rows `40955`
- `ou_funding_z`: max diff `6.0`, diff rows `100354`
- `crowding_pressure`: max diff `1.66916939`, diff rows `105064`

Old 2026 candidate versus stricter clean candidate:

- `last_funding_rate`: max diff `0.0003818`, diff rows `16861`
- `funding_pressure`: max diff `0.0449136`, diff rows `16897`
- `funding_z_score`: max diff `17.65945418`, diff rows `16897`
- `funding_price_divergence`: max diff `6.0`, diff rows `2917`
- `ou_funding_z`: max diff `6.0`, diff rows `16878`

## Fixes Applied

- `scripts/build_alpha7_01965_cleanfunding_candidates_20260529.py`
  - overlays `FUNDING_DERIVED_COLS` from `data/splits/year_oos/training_features_2026_rebuilt.csv` for 2026 eval;
  - uses clean 2025 unified frame for 2025 train;
  - limits M7 overlay to `m7_*` plus clean `sig_ai_squeeze`, so M7 frames cannot overwrite base funding-derived columns;
  - keeps fail-fast timestamp, duplicate, forbidden regime, v2 regime, and `regime4_pred_*` checks.
- `trading_bot_modules/binance_live_fetcher.py`
  - removed `bfill()` after backward funding merge;
  - remaining missing values now fail after causal `ffill()` instead of being filled from future data.
- `scripts/train_eval_alpha7_directional_dsac_router_20260529.py`
  - default train/eval CSVs now point to cleanfunding `01965` candidate frames, with explicit env overrides for isolated historical reproduction.

## Clean Validation

After the stricter builder fix, the 2026 clean candidate frame matches `data/splits/year_oos/training_features_2026_rebuilt.csv` on the funding-derived family:

- exact zero diff for `last_funding_rate`, `funding_abs`, `funding_roc_*`, `funding_z_score`, `funding_price_divergence`, `long_squeeze_risk`, `short_squeeze_risk`, `mta_funding`, `ou_funding_z`, and `crowding_pressure`;
- `squeeze_power` max diff is only CSV float roundtrip noise: `8.73e-11`.

## Retest Impact

Clean Alpha7 `01965` validation-best is:

- `primary_no_tp_fallback_v2`
- validation Cost3: `42.73%` PnL / `-29.52%` MDD / `90` trades / `11.11%` WR
- OOS Cost3: `43.95%` PnL / `-32.23%` MDD / `76` trades / `18.42%` WR

The highest reported OOS combination is `primary_v2_fallback_no_tp` at `55.30%` PnL, but it is not the validation-selected baseline.

Clean Alpha8 Mamba DSAC retest:

- validation-selected fixed template: `fixed_60_aggressive`
- validation: `91.83%` PnL / `-36.38%` MDD / `100` trades / `46.00%` WR
- OOS: `65.81%` PnL / `-26.94%` MDD / `113` trades / `51.33%` WR
- learned Mamba DSAC actor OOS: `8.85%` PnL / `-32.81%` MDD / `76` trades / `43.42%` WR
- fixed 54/55 high-cap collapses to OOS `-0.67%` after the stricter funding-derived overlay.

## Promotion Decision

Do not promote or compose from any stale Alpha7/Alpha8 artifact that lacks clean funding provenance. The clean Alpha8 retest is research-only and not live-wired. Live promotion requires changing the active trading bot artifact path to a clean-funded artifact and re-running full runtime-native parity with the clean manifest.

## Artifacts

- `/home/llewyn/crypto-scalping/tmp/causal_regen_20260516/alpha7_01965_cleanfunding_candidates_20260529/candidate_cleanfunding_audit.json`
- `/home/llewyn/crypto-scalping/tmp/causal_regen_20260516/alpha7_01965_cleanfunding_candidates_20260529/tp_sl_path_edge_feature_audit.json`
- `/home/llewyn/crypto-scalping/tmp/causal_regen_20260516/alpha7_submodel_01965_cleanfunding_v1_20260529/report.json`
- `/home/llewyn/crypto-scalping/tmp/causal_regen_20260516/alpha8_highcap_mamba_seq_dsac_risk_cleanfunding_20260529/summary.json`
- `/home/llewyn/crypto-scalping/tmp/causal_regen_20260516/alpha8_highcap_mamba_seq_dsac_risk_cleanfunding_20260529/grid.csv`
