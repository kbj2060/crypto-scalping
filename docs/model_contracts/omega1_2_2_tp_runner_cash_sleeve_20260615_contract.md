# Omega1.2.2 TP Runner Cash Sleeve 20260615

## Status

`redteam_pass_shadow_candidate_not_live_wired`

This is the next named Omega version after
`omega1_2_1_tp_runner_clean_repair_20260613`. It preserves the Omega1.2.1
TP-runner clean-repair primary and adds a cash-only sleeve trained on Omega
state features.

## Purpose

Transfer the Alpha parent-CASH sleeve pattern into the clean Omega TP-runner
lineage without adding legacy aliases or fallback feature compatibility:

- primary model remains `omega1_2_1_tp_runner_clean_repair_20260613`;
- sleeve may enter only when the primary parent decision is CASH and no primary
  position is open;
- if primary becomes active while the sleeve is open, the sleeve exits via
  `fallback_primary_takeover`;
- OOS is reported after validation-only candidate ranking;
- Red Team PASS excludes PnL/OOS lift and checks only logical, data, feature,
  and contract blockers.

## Runtime Contract

- `model_id`: `omega1_2_2_tp_runner_cash_sleeve_20260615`
- base model: `omega1_2_1_tp_runner_clean_repair_20260613`
- primary TP-runner mode: `mom3_quality`
- `quality_min`: `0.70`
- `extend_mult`: `1.75`
- `floor_frac`: `0.75`
- `max_extensions`: `2`
- sleeve model: `HistGradientBoostingClassifier`
- sleeve risk: `mid_tp030_sl018_n055_h192`
- sleeve `min_edge`: `0.004`
- sleeve threshold: `0.55`

## Feature Contract

The sleeve is retrained on the existing Omega payload state features from
`eval_omega1_2_1_tp_runner_20260610._build()`, plus only these primary-state
derived features:

- `primary_is_cash`
- `primary_active_roll_12`
- `primary_active_roll_48`
- `primary_cash_streak`

Forbidden active sleeve features:

- `tp_sl_action_score`
- `teacher_*`
- `regime4_pred_*`
- `clean_regime4_*`
- `clean_regime_2024_unsup_v4_*`

The implementation is fail-fast if any forbidden feature appears.

## Red Team Result

- status: `redteam_pass_shadow_candidate`
- `redteam_pass`: `true`
- `redteam_blockers`: `[]`
- pass policy: PnL and OOS lift are diagnostics only. FAIL is limited to
  logical defects, data/feature contract violations, forbidden feature leakage,
  missing train/test CASH rows, or failed sleeve candidate generation.

## Metrics

Omega1.2.1 clean-repair baseline replay:

- validation: `+160.22%` PnL, MDD `-27.64%`, WR `59.46%`, trades `37`
- OOS: `+85.70%` PnL, MDD `-15.64%`, WR `66.67%`, trades `18`

Omega1.2.2 selected cash sleeve:

- validation: `+172.46%` PnL, MDD `-26.54%`, WR `60.87%`, trades `46`,
  fallback entries `9`
- OOS: `+86.26%` PnL, MDD `-15.64%`, WR `60.00%`, trades `40`,
  fallback entries `22`

These metrics are diagnostics and are not part of the Red Team PASS/FAIL gate.

## Artifacts

- script:
  `scripts/train_eval_omega1_2_2_tp_runner_cash_sleeve_20260615.py`
- report:
  `tmp/causal_regen_20260516/omega1_2_2_tp_runner_cash_sleeve_20260615/report.json`
- ranking:
  `tmp/causal_regen_20260516/omega1_2_2_tp_runner_cash_sleeve_20260615/validation_only_ranking.csv`
- manifest:
  `data/ensemble/supervised/omega1_2_2_tp_runner_cash_sleeve_20260615/candidate_manifest.json`

## Live Wiring

This contract names and propagates the next Omega version to the project
subagents. It does not wire the model into `trading_bot.py` live execution.
Live wiring requires a separate implementation change and runtime parity check.
