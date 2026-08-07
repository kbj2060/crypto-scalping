# Deep Stop CD18 Funding Bug Audit - 2026-05-29

## Verdict

`alpha7_submodel_01965_decontam_deep_stop_cd18_20260528` is affected by the funding feature bug.

The active trading bot default still points to the stale pre-clean-funding artifact family:

- `data/ensemble/supervised/alpha7_submodel_01965_decontam_v2_tp_20260528`

That directory is already marked:

- `DEPRECATED_DO_NOT_USE.json`
- `alpha7_live_manifest.json` status: `deprecated_do_not_use_active_or_candidate`

Do not use this lineage as active runtime, candidate baseline, parent block, fallback block, TP/SL sidecar source, or promotion evidence.

## Runtime Wiring Finding

`trading_bot.py` defaults reference the stale artifact family:

- primary parent: `data/ensemble/supervised/alpha7_submodel_01965_decontam_v2_tp_20260528/primary_parent.pkl`
- fallback parent: `data/ensemble/supervised/alpha7_submodel_01965_decontam_v2_tp_20260528/fallback_alpha43_no_legacy_parent.pkl`
- TP/SL sidecar: `data/ensemble/supervised/alpha7_submodel_01965_decontam_v2_tp_20260528/tp_sl_path_edge_predictor.pkl`
- runtime config: `data/ensemble/supervised/alpha7_submodel_01965_decontam_v2_tp_20260528/alpha7_decontam_deep_stop_cd18_runtime_config.json`

The currently running trading bot process was also observed as `trading_bot.py`; with no cleanfunding override in the checked environment output, it follows those code defaults unless externally wrapped elsewhere.

## Layer Exposure

Funding-family inputs are present in the active/stale Alpha7 parent and fallback artifacts:

- active primary: `funding_abs`, `funding_pressure`, `funding_price_divergence`, `long_squeeze_risk`, `crowding_pressure`, `squeeze_power`
- active fallback: `funding_abs`, `funding_pressure`, `funding_price_divergence`, `long_squeeze_risk`, `crowding_pressure`, `squeeze_power`

The Alpha3/V31 cash-fallthrough stack used by `deep_stop_cd18` is also exposed:

- `deep_scout_state24_v2.pt`: `last_funding_rate`, `long_squeeze_risk`, `funding_price_divergence`
- `teacher_state24_v2.pt`: `funding_abs`, `funding_pressure`, `funding_price_divergence`, `long_squeeze_risk`, `crowding_pressure`, `squeeze_power`
- `parent_state24_v2.pkl`: `funding_abs`, `funding_pressure`, `funding_price_divergence`, `long_squeeze_risk`, `crowding_pressure`, `squeeze_power`
- `v21_runner_state24_v2.pkl` cost runner: `funding_abs`, `funding_pressure`, `funding_price_divergence`, `long_squeeze_risk`, `crowding_pressure`, `squeeze_power`

Therefore this is not isolated to one feature or one sub-layer.

## Quantified Candidate Frame Difference

Compared stale Alpha7 candidate frames against the cleanfunding rebuild:

Stale source:
- `tmp/causal_regen_20260516/alpha7_1_01965_v2only_tp_sl_action_score_20260528/`

Clean rebuild:
- `tmp/causal_regen_20260516/alpha7_01965_cleanfunding_candidates_20260529/`

2025:

- `last_funding_rate`: diff rows `105064`, max abs diff `0.00140792`
- `funding_abs`: diff rows `105064`, max abs diff `0.00138089`
- `funding_pressure`: diff rows `105064`, max abs diff `0.20464266`
- `crowding_pressure`: diff rows `105064`, max abs diff `1.669169391`
- `squeeze_power`: diff rows `105055`, max abs diff `8348629.444`
- `long_squeeze_risk`: diff rows `97724`, max abs diff `0.3572492751`
- `funding_roc_288`: diff rows `96814`, max abs diff `7.283371663`
- `ou_funding_z`: diff rows `100354`, max abs diff `6.0`

2026:

- `funding_pressure`: diff rows `16897`, max abs diff `0.0449136`
- `funding_z_score`: diff rows `16897`, max abs diff `17.65945418`
- `crowding_pressure`: diff rows `16897`, max abs diff `1.50420161`
- `funding_roc_288`: diff rows `16882`, max abs diff `4.413858614`
- `ou_funding_z`: diff rows `16878`, max abs diff `6.0`
- `squeeze_power`: diff rows `16862`, max abs diff `2150112.479`
- `last_funding_rate`: diff rows `16861`, max abs diff `0.0003818`
- `funding_abs`: diff rows `16861`, max abs diff `0.00034872`

## Clean Replacement State

Clean research artifact exists:

- `data/ensemble/supervised/alpha7_submodel_01965_cleanfunding_v1_20260529`

It is explicitly marked:

- status: `research_only_not_live_wired`

Its validation-selected result is materially weaker than stale `deep_stop_cd18`, so it should not be silently promoted:

- clean Alpha7 OOS Cost3: `43.95%` PnL, `-32.23%` MDD, `76` trades, `18.42%` WR
- stale `deep_stop_cd18` OOS Cost3: `198.78%` PnL, `-18.22%` MDD, `109` trades, `44.04%` WR

This confirms that part of the old performance depends on the stale funding-derived feature lineage and cannot be used as clean promotion evidence.

## Decision

Treat `deep_stop_cd18` as funding-bug affected and not clean for active/live promotion.

Required remediation:

1. Do not continue active/live runtime on `alpha7_submodel_01965_decontam_v2_tp_20260528`.
2. Retrain or redesign from cleanfunding candidate frames.
3. Re-run runtime-native parity and precision retest from clean artifacts only.
4. Add a live startup fail-fast guard that refuses deprecated artifact directories.
5. Do not use stale `deep_stop_cd18` metrics as baseline for new Alpha8/Alpha7 promotion.
