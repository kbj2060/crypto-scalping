# Alpha4 New-Feature Full Retrain Contract - 2026-05-17

## Goal

Retrain the Alpha3 stack as Alpha4 on the causal/new feature frame generated under
`tmp/causal_regen_20260516`, after removing the legacy AI alias, anchor, and
TimesNet leak-risk features.

## Inputs

- Train frame: `tmp/causal_regen_20260516/ai_feature_combo_grid/trade_candidates_2025_patchtst__tide__dlinear.csv`
- OOS frame: `tmp/causal_regen_20260516/ai_feature_combo_grid/trade_candidates_2026_patchtst__tide__dlinear.csv`
- Selection window: 2025-10-01 through 2025-12-31
- OOS window: fixed 2026 frame only after selection
- Execution contract: corrected Alpha3 close-fallback limit execution

## Retrained Layers

- Parent: HGB governor, trained from the causal/new feature contract.
- V27-style deep utility: TCN sequence model retrained on the causal/new frame.
- Teacher gate: sequence teacher retrained to follow the Alpha4 parent decisions.
- V21.2 runner: cost/jackpot add-on runner retrained on Alpha4 final decisions.
- V31/exit overlay: evaluated under the Alpha3 corrected limit-close contract; a
  follow-up overlay ablation showed the selected safe variant disables deep sleeve.

## Removed Legacy Features

- `patchtst_pred`
- `patchtst_confidence`
- `ai_anchor_revert_prob`
- `ai_anchor_overheat`
- `ai_anchor_trend_escape_prob`
- `timesnet_cycle_sin`
- `timesnet_cycle_cos`
- `timesnet_cycle_delta`

## Primary Run

Command:

```bash
venv/bin/python scripts/eval_alpha4_new_features_full_retrain_20260517.py --only hgb --teacher-epochs 35 --v27-epochs 80 --stride 6
```

Artifacts:

- Report: `tmp/causal_regen_20260516/alpha4_new_features_full_retrain_20260517/alpha4_new_features_full_retrain_summary.json`
- Audit: `tmp/causal_regen_20260516/alpha4_new_features_full_retrain_20260517/alpha4_new_features_full_retrain_audit.json`
- Grid: `tmp/causal_regen_20260516/alpha4_new_features_full_retrain_20260517/alpha4_new_features_full_retrain_grid.csv`
- Parent: `tmp/causal_regen_20260516/alpha4_new_features_full_retrain_20260517/artifacts/hgb/parent.pkl`
- Teacher: `tmp/causal_regen_20260516/alpha4_new_features_full_retrain_20260517/artifacts/hgb/teacher_gate.pt`
- Runner: `tmp/causal_regen_20260516/alpha4_new_features_full_retrain_20260517/artifacts/hgb/runner.pkl`
- V27-style utility: `tmp/causal_regen_20260516/alpha4_new_features_full_retrain_20260517/artifacts/alpha4_v27_style_deep_alpha_utility/alpha4_v27_style_deep_alpha_utility.pt`

Primary 2026 OOS result:

| Candidate | Cost1 PnL | Cost1 MDD | Cost2 PnL | Cost3 PnL | Trades | Deep Entries |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Alpha3 frozen reference on causal frame | +133.96% | -33.70% | +77.09% | +37.94% | 376 | 305 |
| Alpha4 HGB full retrain | -26.19% | -52.57% | -27.74% | -37.71% | 185 | 147 |

Selected Alpha4 runtime:

- Teacher runtime: `noflip_c0.68_parent_scale0.85`
- Runner config: `v21_2_parent_noop`
- Audit verdict: `iterate`

## Failure Analysis

The primary Alpha4 full retrain should not be promoted. The direct failure source
is the newly trained V27-style deep sleeve:

- OOS deep entries: 147
- Deep stop-loss exits: 112
- Deep take-profit exits: 10
- Deep max-hold exits: 24

This means the new V27-style utility model is over-admitting deep-alpha trades
under the inherited V31 threshold contract. It predicts enough positive utility
to trigger entries, but the realized exit distribution is dominated by stop-loss.

## Ablations

Deep sleeve disabled, same Alpha4 parent/teacher/runner:

| Variant | Cost1 PnL | Cost1 MDD | Cost2 PnL | Cost3 PnL | Trades | Deep Entries |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Alpha4 parent/teacher only | +31.89% | -38.00% | +32.59% | +30.25% | 43 | 0 |

Overlay reselection on 2025Q4 selected `alpha4_no_deep_parent_only`; OOS matched
the no-deep ablation. This confirms the retrained V27-style sleeve is not usable
without stricter calibration or a different target/reward.

## Next Constraints

Before another Alpha4 attempt:

- Do not promote the current Alpha4 full retrain.
- Calibrate V27 utility probabilities/thresholds before enabling deep sleeve.
- Add an explicit no-deep fallback candidate to all future overlay selection grids.
- Consider training V27 targets on realized TP/SL path outcomes rather than max
  horizon utility, because current labels do not punish stop-loss-dominated paths
  strongly enough.
