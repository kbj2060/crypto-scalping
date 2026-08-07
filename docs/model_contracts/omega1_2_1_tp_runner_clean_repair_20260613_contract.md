# Omega1.2.1 TP Runner Clean Repair 20260613

## Status

`clean_repair_candidate_shadow_required`

This model replaces the deprecated `omega1_2_1_tp_runner_only_baseline_20260612` active TP-runner path. The old `+205.92%` OOS result remains deprecated because it was selected with OOS feedback and used non-equivalent replay assumptions.

## Purpose

Restore the TP-runner baseline under clean evaluation rules:

- select TP-runner parameters using validation data only;
- report OOS only once after validation selection;
- remove active dependency on the OOS-mined TP-runner meta-selector bundle;
- use next-open taker entry and intrabar high/low price-barrier exits in the repair audit;
- keep fail-fast feature contracts and no legacy aliases.

## Runtime Contract

- `model_id`: `omega1_2_1_tp_runner_clean_repair_20260613`
- parent entry/risk owner: Omega1.2.1 true-leverage parent
- TP-runner mode: `mom3_quality`
- `quality_min`: `0.70`
- `momentum_min`: `0.0`
- `extend_mult`: `1.75`
- `floor_frac`: `0.75`
- `max_extensions`: `2`
- active meta-selector bundle: not used

## Selection Policy

The selected runner configuration was chosen by `selection_score_val_only` on validation data. OOS metrics were computed after selection and were not used for ranking.

## Clean Repair Metrics

Validation:

- PnL: `+160.22%`
- MDD: `-27.64%`
- WR: `59.46%`
- trades: `37`

OOS:

- PnL: `+85.70%`
- MDD: `-15.64%`
- WR: `66.67%`
- trades: `18`

Clean no-runner accounting baseline for context:

- validation: `+49.16%`, MDD `-33.16%`, WR `46.67%`, trades `45`
- OOS: `+120.07%`, MDD `-15.64%`, WR `65.00%`, trades `20`

The clean selected TP-runner improves validation substantially but does not beat the clean no-runner baseline on OOS. Treat this as a shadow-required candidate, not proven live alpha.

## Artifacts

- report: `tmp/causal_regen_20260516/omega1_2_1_tp_runner_clean_repair_20260613/report.json`
- ranking: `tmp/causal_regen_20260516/omega1_2_1_tp_runner_clean_repair_20260613/validation_only_ranking.csv`
- manifest: `data/ensemble/supervised/omega1_2_1_tp_runner_clean_repair_20260613/baseline_manifest.json`

## Audit Notes

- No forbidden feature prefixes are introduced by this repair path.
- OOS-mined `tp_runner_meta_selector_20260610` is not part of active TP-runner decisions.
- Existing deprecated artifacts remain blocked for promotion evidence.
